import argparse
import json
import os
from pprint import pprint
from typing import Dict, List, Union

import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizer

from utils import get_span_indices, load_processed_dataset

parser = argparse.ArgumentParser(description="Configuration for template generation")
parser.add_argument("--dataset", type=str, default="dd", choices=["dd", "persona"])
parser.add_argument(
    "--lm_type",
    type=str,
    default="bert-ft",
    choices=["bert-ft", "bert-scratch", "roberta-ft", "roberta-scratch"],
)
parser.add_argument(
    "--scoring_method",
    type=str,
    default="cr_span",
    choices=["response", "contextresponse", "salience", "cr_span"],
)

parser.add_argument(
    "--data_path",
    type=str,
    default="./data/processed/dd/{}.jsonl",
    choices=[
        "./data/processed/dd/{}.jsonl",
        "./data/processed/persona/{}.jsonl",
    ],
)
parser.add_argument(
    "--max_span_length",
    type=int,
    default=5,
)
parser.add_argument("--threshold", type=float, default=0)

args = parser.parse_args()


def main():
    device = torch.device("cuda")

    """
    Load the LM for scoring
    """
    if args.lm_type == "bert-ft":
        lm_path = "./model/lm/" + args.dataset + "_ft/"
        model = BertForMaskedLM.from_pretrained(lm_path)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif args.lm_type == "bert-scratch":
        model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    else:
        raise NotImplementedError
    model.to(device)

    train_dataset = load_processed_dataset(args.data_path.format("train"))
    valid_dataset = load_processed_dataset(args.data_path.format("valid"))
    test_dataset = load_processed_dataset(args.data_path.format("test"))

    for idx, dataset in enumerate([valid_dataset, train_dataset, test_dataset]):
        setname = ["valid", "train", "test"][idx]
        with torch.no_grad():
            for idx, sample in enumerate(dataset):
                print(f"{idx}/{len(dataset)}")
                if idx == 100:
                    break
                context = " ".join(sample["context"])
                tokenized_reply = (
                    ["[CLS]"] + tokenizer.tokenize(sample["reply"]) + ["[SEP]"]
                )
                reply = torch.tensor(
                    tokenizer.convert_tokens_to_ids(tokenized_reply[:])
                )

                token_order, score_list = template_generation(
                    context,
                    reply,
                    model,
                    tokenizer,
                    device,
                    args.scoring_method,
                    contain_persona=True if args.dataset == "persona" else False,
                    masked_token_index_in_response_list=[],
                    score_list_buffer=[],
                )

                assert len(tokenized_reply) - 2 == len(token_order) == len(score_list)
                assert all([len(el) == len(token_order) for el in score_list])
                dataset[idx]["tokenized_reply"] = tokenized_reply[1:-1]
                dataset[idx]["mask_order"] = [int(el) for el in token_order]
                dataset[idx]["score_list"] = score_list

        os.makedirs(
            "./score/{}/{}/".format(args.scoring_method, args.dataset), exist_ok=True
        )
        with open(
            "./score/{}/{}/{}.jsonl".format(args.scoring_method, args.dataset, setname),
            "w",
        ) as f:
            for line in dataset:
                json.dump(line, f)
                f.write("\n")


def template_generation(
    context: str,
    encoded_response: torch.Tensor,
    model: BertForMaskedLM,
    tokenizer: BertTokenizer,
    device: torch.device,
    scoring_method: str,
    contain_persona: bool,
    masked_token_index_in_response_list: List[int] = [],
    score_list_buffer: List[List[float]] = [],
):
    assert len(masked_token_index_in_response_list) == len(score_list_buffer)
    softmax = torch.nn.Softmax(dim=2)
    mask_token_id = tokenizer.mask_token_id
    encoded_context = tokenizer(context, return_tensors="pt")["input_ids"][0]
    encoded_conversation = torch.cat([encoded_context[:-1], encoded_response[1:]])
    original_token_list = encoded_response[1:-1]

    # Get every possible span indices
    span_candidate_indices = get_span_indices(
        len(encoded_response) - 2, args.max_span_length
    )
    copied_conversation = encoded_conversation.repeat(len(span_candidate_indices), 1)

    original_span_list = []
    for idx, span in enumerate(span_candidate_indices):
        beg, end = [el + len(encoded_context) - 1 for el in span]
        original_span_list.append(
            copied_conversation[idx, [range(beg, end)]][0].numpy().tolist()
        )
        copied_conversation[idx, [range(beg, end)]] = mask_token_id
        span_candidate_indices[idx] = [beg, end]

    conversation_logit = (
        softmax(model(copied_conversation.to(device), return_dict=True).logits)
        .detach()
        .cpu()
        .numpy()
    )

    score_list = []
    for span_index, spans in enumerate(original_span_list):
        beg_idx, end_idx = span_candidate_indices[span_index]
        original_span_ids = spans

        # shape of [span_legnth, vocab_size]
        span_vocab_score = conversation_logit[span_index][beg_idx:end_idx]
        assert len(span_vocab_score) == len(original_span_ids)
        score = []
        for vocab_idx, vocab_score in enumerate(span_vocab_score):
            score.append(vocab_score[original_span_ids[vocab_idx]])
        score_list.append(
            {
                "span_index": [beg_idx, end_idx],
                "span_words": original_span_ids[:],
                "score": score[:],
            }
        )
    print(score_list)
    input()
    for dim1 in range(len(original_token_list)):
        if dim1 in masked_token_index_in_response_list:
            score_list.append(-10000.0)
            continue
        row, res_col, conv_col, original_token = (
            dim1,
            dim1 + 1,
            len(encoded_context) - 1 + dim1,
            original_token_list[dim1],
        )
        resscore = response_logit[row][res_col][original_token]
        if scoring_method != "response":
            convscore = conversation_logit[row][conv_col][original_token]

        if scoring_method == "response":
            score_list.append(float(resscore))
        elif scoring_method == "contextresponse":
            score_list.append(float(convscore))
        elif scoring_method == "salience":
            score_list.append(float(convscore) - float(resscore))

    sorted_index = [
        el
        for el in np.argsort(score_list)
        if el not in masked_token_index_in_response_list
    ]
    masked_token_index = sorted_index[0]
    encoded_response[masked_token_index + 1] = mask_token_id
    assert masked_token_index < len(encoded_response) - 2

    masked_token_index_in_response_list.append(masked_token_index)
    score_list_buffer.append(score_list[:])

    return template_generation(
        context,
        encoded_response,
        model,
        tokenizer,
        device,
        scoring_method,
        contain_persona,
        masked_token_index_in_response_list,
        score_list_buffer,
    )


if __name__ == "__main__":
    main()
