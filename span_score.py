import os, json
import numpy as np

import argparse

from pprint import pprint
from typing import Dict, List, Union


import torch
from transformers import BertForMaskedLM, BertTokenizer

from utils import (
    load_processed_dataset,
    get_processed_dataset,
    match_retrieved_response,
    get_span_indices,
)

"""
Hierarchical Masking Strategy with Recursive
"""

parser = argparse.ArgumentParser(
    description="Configuration for template generation"
)
parser.add_argument(
    "--dataset", type=str, default="dd", choices=["dd", "persona"]
)
parser.add_argument("--max_span_length", type=int, default=4)
parser.add_argument(
    "--data_path",
    type=str,
    default="./data/processed/{}/{}.jsonl",
)

parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument(
    "--decode_strategy",
    type=str,
    default="argmax_onestep",
    choices=["argmaxSequential"],
)

parser.add_argument(
    "--similarity",
    type=str,
    default="qq",
    choices=["qr", "qq"],
)


args = parser.parse_args()


def main():
    device = torch.device("cuda")

    """
    Load the LMs for scoring and generation
    """

    lm_path = "./logs/MaskResBert-{}/model/".format(args.dataset)
    context_model = BertForMaskedLM.from_pretrained(lm_path)
    context_model.to(device)

    context_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    context_tokenizer.add_tokens(
        [
            "[PERSONA1]",
            "[PERSONA2]",
            "[SEPT]",
        ]
    )
    if context_tokenizer.pad_token is None:
        context_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        context_tokenizer.pad_token = "[PAD]"

    softmax = torch.nn.Softmax(dim=2)

    train_dataset = load_processed_dataset(
        args.data_path.format(args.dataset, "train")
    )
    valid_dataset = load_processed_dataset(
        args.data_path.format(args.dataset, "valid")
    )
    test_dataset = load_processed_dataset(
        args.data_path.format(args.dataset, "test")
    )

    valid_dataset = match_retrieved_response(
        valid_dataset,
        "./data/repr/{}/valid_top_sorted.jsonl".format(args.dataset),
        train_dataset,
        setup=args.similarity,
    )
    test_dataset = match_retrieved_response(
        test_dataset,
        "./data/repr/{}/test_top_sorted.jsonl".format(args.dataset),
        train_dataset,
        setup=args.similarity,
    )

    for dataset_idx, dataset in enumerate(
        [test_dataset]
    ):  # , valid_dataset]):
        final_output_list = []
        with torch.no_grad():
            for sample_index, sample in enumerate(dataset):

                print(sample_index)
                context, golden_response, retrieved_response = (
                    sample["context"],
                    sample["reply"],
                    sample["retrieved"],
                )
                final_output_bag = {
                    "context": context,
                    "golden": golden_response,
                    "retrieved": retrieved_response,
                }

                context = "[SEPT]".join(sample["context"]) + "[SEPT]"
                if args.dataset == "persona":
                    your_persona = (
                        "[PERSONA1]".join(sample["your_persona"])
                        + "[PERSONA1]"
                    )
                    partner_persona = (
                        "[PERSONA2]".join(sample["parter_persona"])
                        + "[PERSONA2]"
                    )
                    final_output_bag["context"] = [
                        your_persona,
                        partner_persona,
                    ] + final_output_bag["context"]
                    context = your_persona + partner_persona + context

                full_conv = context + retrieved_response
                encoded_conv = context_tokenizer(full_conv)["input_ids"]
                if len(encoded_conv) > 500:
                    continue
                encoded_response = context_tokenizer(retrieved_response)[
                    "input_ids"
                ]
                len_response = len(encoded_response) - 2

                assert (
                    encoded_response[1:]
                    == encoded_conv[-len(encoded_response[1:]) :]
                )
                tokenized_response = context_tokenizer.tokenize(
                    retrieved_response
                )
                span_indices = get_span_indices(
                    len_response, args.max_span_length
                )
                conv4score = [
                    encoded_conv[:] for _ in range(len(span_indices))
                ]

                """
                MASKING
                """
                original_token_id_list = []

                for span_index, (beg_index, end_index) in enumerate(
                    span_indices
                ):
                    conv_origin_ids = conv4score[span_index][
                        -len_response
                        - 1
                        + beg_index : -len_response
                        - 1
                        + end_index
                    ][:]
                    for tok_idx_in_span in range(end_index - beg_index):
                        assert (
                            conv_origin_ids[tok_idx_in_span]
                            == conv4score[span_index][
                                beg_index + tok_idx_in_span - len_response - 1
                            ]
                        )
                        conv4score[span_index][
                            beg_index + tok_idx_in_span - len_response - 1
                        ] = context_tokenizer.mask_token_id
                    original_token_id_list.append(conv_origin_ids)

                # [span_num, conversation_len + 2, vocab]
                try:
                    conv_output = (
                        softmax(
                            context_model(
                                torch.tensor(conv4score).to(device),
                                return_dict=True,
                            ).logits
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                except:
                    final_output_list.append({})
                    continue

                assert (
                    len(conv_output)
                    == len(original_token_id_list)
                    == len(span_indices)
                )

                span_score = []
                for span_order, span_index in enumerate(span_indices):
                    beg_idx, end_idx = (
                        span_index[0]
                        + len(conv_output[0])
                        - 1
                        - len_response,
                        span_index[1]
                        + len(conv_output[0])
                        - 1
                        - len_response,
                    )
                    score_list = []
                    for tok_idx_in_span in range(end_idx - beg_idx):
                        score_list.append(
                            float(
                                conv_output[span_order][
                                    beg_idx + tok_idx_in_span
                                ][
                                    original_token_id_list[span_order][
                                        tok_idx_in_span
                                    ]
                                ]
                            )
                        )
                    span_score.append(score_list)
                final_output_bag["span_indices"] = span_indices
                final_output_bag["span_score"] = span_score
                final_output_bag["tokenized_response"] = tokenized_response
                final_output_list.append(final_output_bag)

        setname = "test" if dataset_idx == 0 else "valid"
        with open(
            "./generated/span{}-{}-{}_{}.jsonl".format(
                args.max_span_length,
                args.dataset,
                setname,
                args.similarity,
            ),
            "w",
        ) as f:
            for line in final_output_list:
                json.dump(line, f)
                f.write("\n")


if __name__ == "__main__":
    main()
