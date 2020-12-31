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
parser.add_argument(
    "--data_path",
    type=str,
    default="./data/processed/{}/{}.jsonl",
)

parser.add_argument("--threshold", type=float, default=0.1)
parser.add_argument(
    "--decode_strategy",
    type=str,
    default="argmax_onestep",
    choices=["argmaxSequential"],
)


args = parser.parse_args()


def main():
    device = torch.device("cuda")

    """
    Load the LMs for scoring and generation
    """
    lm_path = "./model/lm/" + args.dataset + "_ft/"
    response_model = BertForMaskedLM.from_pretrained(lm_path)
    lm_path = "./logs/MaskResBert-{}/model/".format(args.dataset)
    context_model = BertForMaskedLM.from_pretrained(lm_path)
    response_model.to(device)
    context_model.to(device)

    context_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
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

    response_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    assert context_tokenizer.mask_token_id == response_tokenizer.mask_token_id
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
    )
    test_dataset = match_retrieved_response(
        test_dataset,
        "./data/repr/{}/test_top_sorted.jsonl".format(args.dataset),
        train_dataset,
    )

    for dataset_idx, dataset in enumerate([valid_dataset, test_dataset]):
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
                    context = your_persona + partner_persona + context

                full_conv = context + retrieved_response
                encoded_conv = context_tokenizer(full_conv)["input_ids"]
                if len(encoded_conv) > 500:
                    continue
                encoded_response = response_tokenizer(retrieved_response)[
                    "input_ids"
                ]
                len_response = len(encoded_response) - 2

                assert (
                    encoded_response[1:]
                    == encoded_conv[-len(encoded_response[1:]) :]
                )

                conv4score = [encoded_conv[:] for _ in range(len_response)]
                response4score = [
                    encoded_response[:] for _ in range(len_response)
                ]

                """
                MASKING
                """
                original_token_id_list = []

                for token_index in range(len_response):
                    conv_origin_id = conv4score[token_index][
                        -len_response - 1 + token_index
                    ]
                    conv4score[token_index][
                        -len_response - 1 + token_index
                    ] = context_tokenizer.mask_token_id
                    response_origin_id = response4score[token_index][
                        1 + token_index
                    ]
                    response4score[token_index][
                        1 + token_index
                    ] = response_tokenizer.mask_token_id
                    assert conv_origin_id == response_origin_id
                    original_token_id_list.append(conv_origin_id)

                """
                Run Model
                """
                # [len_res, len_res + len_context + 2, vocab]
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
                # [len_res, len_res + 2, vocab]
                res_output = (
                    softmax(
                        response_model(
                            torch.tensor(response4score).to(device),
                            return_dict=True,
                        ).logits
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                assert len_response == len(original_token_id_list)

                conv_score, res_score = [], []
                for score_token_index in range(len_response):
                    conv_col = (
                        len(conv_output[0])
                        - len_response
                        - 1
                        + score_token_index
                    )
                    res_col = 1 + score_token_index
                    original_tok_id = original_token_id_list[
                        score_token_index
                    ]

                    conv_score.append(
                        conv_output[score_token_index][conv_col][
                            original_tok_id
                        ]
                    )
                    res_score.append(
                        res_output[score_token_index][res_col][
                            original_tok_id
                        ]
                    )
                assert (
                    len(conv_score)
                    == len(res_score)
                    == len_response
                    == len(encoded_response) - 2
                )
                final_output_bag["context_score"] = [
                    float(el) for el in conv_score
                ]
                final_output_bag["response_score"] = [
                    float(el) for el in res_score
                ]
                final_output_bag["encoded_retrieved_response"] = [
                    int(el) for el in encoded_response[1:-1]
                ]

                assert context_tokenizer.convert_ids_to_tokens(
                    encoded_response
                ) == response_tokenizer.convert_ids_to_tokens(
                    encoded_response
                )
                tokenized_response = response_tokenizer.convert_ids_to_tokens(
                    encoded_response[1:-1]
                )
                """
                Generation GOGO
                """
                tmp4res = [
                    el
                    if res_score[idx] > args.threshold
                    else response_tokenizer.mask_token
                    for idx, el in enumerate(tokenized_response)
                ]
                mask_index_list_w_res = [
                    idx
                    for idx, el in enumerate(res_score)
                    if el <= args.threshold
                ]

                tmp4conv = [
                    el
                    if conv_score[idx] > args.threshold
                    else context_tokenizer.mask_token
                    for idx, el in enumerate(tokenized_response)
                ]
                mask_index_list_w_conv = [
                    idx
                    for idx, el in enumerate(conv_score)
                    if el <= args.threshold
                ]
                assert isinstance(context, str)

                input_seq_w_res_score = context + " ".join(tmp4res).strip()
                input_seq_w_conv_score = context + " ".join(tmp4conv).strip()

                def generate(
                    input_sequence: str,
                    model: BertForMaskedLM,
                    mask_index_list,
                    tokenizer,
                ):
                    softmax = torch.nn.Softmax(dim=1)
                    mask_index_list = mask_index_list[:]
                    bert_input_seq = tokenizer(
                        input_sequence, return_tensors="pt"
                    )["input_ids"].to(device)
                    sequence = []
                    step = 0
                    while True:
                        if len(mask_index_list) == 0:
                            break

                        output = model(bert_input_seq, return_dict=True)[
                            "logits"
                        ][0][-1 - len_response : -1]
                        output = softmax(output)
                        max_output = torch.max(output, 1)
                        # Max prob of each timestep
                        max_prob = max_output.values.detach().cpu().numpy()
                        # Most likely word for each timestep
                        max_word = max_output.indices.detach().cpu().numpy()
                        max_prob = [
                            -100
                            if idx not in mask_index_list
                            else max_prob[idx]
                            for idx in range(len(tokenized_response))
                        ]

                        max_word_index = np.argmax(max_prob)
                        selected_word = max_word[max_word_index]

                        sequence.append(
                            [
                                int(max_word_index),
                                tokenizer._convert_id_to_token(selected_word),
                            ]
                        )
                        bert_input_seq[0][
                            -len(tokenized_response) - 1 + max_word_index
                        ] = selected_word
                        mask_index_list.remove(int(max_word_index))
                        step += 1

                    new_reply = " ".join(
                        tokenizer.convert_ids_to_tokens(
                            bert_input_seq[0][
                                -len(tokenized_response) - 1 : -1
                            ]
                        )
                    )
                    return (
                        new_reply,
                        sequence,
                        [
                            int(el)
                            for el in list(
                                bert_input_seq[0][
                                    -len(tokenized_response) - 1 : -1
                                ]
                                .detach()
                                .cpu()
                                .numpy()
                            )
                        ],
                    )

                result_w_res_score = generate(
                    input_seq_w_res_score,
                    context_model,
                    mask_index_list_w_res,
                    context_tokenizer,
                )
                result_w_con_score = generate(
                    input_seq_w_conv_score,
                    context_model,
                    mask_index_list_w_conv,
                    context_tokenizer,
                )
                final_output_bag["context_output"] = result_w_con_score
                final_output_bag["response_output"] = result_w_res_score
                final_output_list.append(final_output_bag)
        setname = "valid" if dataset_idx == 0 else "test"
        with open(
            "./generated/{}-thres{}-decode_{}_{}.jsonl".format(
                args.dataset, args.threshold, args.decode_strategy, setname
            ),
            "w",
        ) as f:
            for line in final_output_list:
                json.dump(line, f)
                f.write("\n")


if __name__ == "__main__":
    main()
