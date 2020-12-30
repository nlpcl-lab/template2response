import argparse
import json
import os
from pprint import pprint
from typing import Dict, List, Union

import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizer

from utils import load_processed_dataset

"""
Hierarchical Masking Strategy with Recursive
"""

parser = argparse.ArgumentParser(
    description="Configuration for template generation"
)
parser.add_argument(
    "--dataset", type=str, default="persona", choices=["dd", "persona"]
)
parser.add_argument(
    "--lm_type",
    type=str,
    default="ftbert",
    choices=["ftbert", "bert-scratch", "roberta-ft", "roberta-scratch"],
)
parser.add_argument(
    "--scoring_method",
    type=str,
    default="response",
    choices=["response", "contextresponse", "salience"],
)

parser.add_argument(
    "--data_path",
    type=str,
    default="./data/processed/persona/{}.jsonl",
    choices=[
        "./data/processed/dd/{}.jsonl",
        "./data/processed/persona/{}.jsonl",
    ],
)
parser.add_argument("--threshold", type=float, default=0)
parser.add_argument(
    "--sequential", type=str, default="1shot", choices=["1shot", "sequential"]
)


args = parser.parse_args()


def main():
    device = torch.device("cuda")

    """
    Load the LM for scoring
    """
    if args.lm_type == "ftbert":
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

    for idx, dataset in enumerate(
        [valid_dataset, train_dataset, test_dataset]
    ):
        setname = ["valid", "test", "train"][idx]
        with torch.no_grad():
            for idx, sample in enumerate(dataset):
                print(f"{idx}/{len(dataset)}")
                context = " ".join(sample["context"])
                tokenized_reply = (
                    ["[CLS]"]
                    + tokenizer.tokenize(sample["reply"])
                    + ["[SEP]"]
                )
                if args.sequential == "sequential":
                    reply = torch.tensor(
                        tokenizer.convert_tokens_to_ids(tokenized_reply[:])
                    )
                    token_order, score_list = template_generation_sequential(
                        context,
                        reply,
                        model,
                        tokenizer,
                        device,
                        args.scoring_method,
                        contain_persona=True
                        if args.dataset == "persona"
                        else False,
                        masked_token_index_in_response_list=[],
                        score_list_buffer=[],
                    )
                    assert (
                        len(tokenized_reply) - 2
                        == len(token_order)
                        == len(score_list)
                    )
                    assert all(
                        [len(el) == len(token_order) for el in score_list]
                    )
                    dataset[idx]["tokenized_reply"] = tokenized_reply[1:-1]
                    dataset[idx]["mask_order"] = [
                        int(el) for el in token_order
                    ]
                    dataset[idx]["score_list"] = score_list
                elif args.sequential == "1shot":
                    try:
                        score_list = template_generation_parallel(
                            context,
                            tokenized_reply,
                            model,
                            tokenizer,
                            device,
                            args.scoring_method,
                            contain_persona=True
                            if args.dataset == "persona"
                            else False,
                        )
                    except Exception as err:
                        print(err)
                        score_list = [
                            -100 for _ in range(len(tokenized_reply) - 2)
                        ]
                    assert len(score_list) == len(tokenized_reply) - 2
                    dataset[idx]["tokenized_reply"] = tokenized_reply[1:-1]
                    dataset[idx]["score_list"] = score_list
                else:
                    raise ValueError

        os.makedirs(
            "./score/{}-{}-{}/{}/".format(
                args.scoring_method,
                args.sequential,
                args.lm_type,
                args.dataset,
            ),
            exist_ok=True,
        )
        output_fname = "./score/{}-{}-{}/{}/{}.jsonl".format(
            args.scoring_method,
            args.sequential,
            args.lm_type,
            args.dataset,
            setname,
        )

        with open(
            output_fname,
            "w",
        ) as f:
            for line in dataset:
                json.dump(line, f)
                f.write("\n")


def template_generation_parallel(
    context: str,
    encoded_response: torch.Tensor,
    model: BertForMaskedLM,
    tokenizer: BertTokenizer,
    device: torch.device,
    scoring_method: str,
    contain_persona: bool,
):
    softmax = torch.nn.Softmax(dim=2)
    mask_token_id = tokenizer.mask_token_id

    encoded_response = torch.tensor(
        tokenizer.convert_tokens_to_ids(encoded_response[:])
    )
    encoded_context = tokenizer(context, return_tensors="pt")["input_ids"][0]
    encoded_conversation = torch.cat(
        [encoded_context[:-1], encoded_response[1:]]
    )
    if len(encoded_conversation) >= 512:
        return [-100 for _ in range(-2 + len(encoded_response))]
    copied_conversation = encoded_conversation.repeat(
        len(encoded_response) - 2, 1
    )
    copied_response = encoded_response.repeat(len(encoded_response) - 2, 1)

    original_token_list = encoded_response[1:-1]
    assert (
        len(original_token_list)
        == len(encoded_response) - 2
        == len(encoded_conversation) - len(encoded_context)
    )

    """
    Mask every token in a response one-by-one
    """
    for tok_idx, tok_id in enumerate(original_token_list):
        copied_response[tok_idx][tok_idx + 1] = mask_token_id
        copied_conversation[tok_idx][
            len(encoded_context) - 1 + tok_idx
        ] = mask_token_id

    # Shape of [# of response_token, # of response_token +2, vocab_size]
    response_logit = (
        softmax(model(copied_response.to(device), return_dict=True).logits)
        .detach()
        .cpu()
        .numpy()
    )
    conversation_logit = (
        softmax(
            model(copied_conversation.to(device), return_dict=True).logits
        )
        .detach()
        .cpu()
        .numpy()
    )

    score_list = []
    for dim1 in range(len(original_token_list)):
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
    return score_list


def template_generation_sequential(
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
    encoded_conversation = torch.cat(
        [encoded_context[:-1], encoded_response[1:]]
    )

    if len(masked_token_index_in_response_list) == len(encoded_response) - 2:
        return masked_token_index_in_response_list, score_list_buffer

    copied_conversation = encoded_conversation.repeat(
        len(encoded_response) - 2, 1
    )
    copied_response = encoded_response.repeat(len(encoded_response) - 2, 1)

    original_token_list = encoded_response[1:-1]
    assert (
        len(original_token_list)
        == len(encoded_response) - 2
        == len(encoded_conversation) - len(encoded_context)
    )

    copied_conversation[
        :, masked_token_index_in_response_list
    ] = mask_token_id
    """
    Mask every token in a response one-by-one
    """
    for tok_idx, tok_id in enumerate(original_token_list):
        if tok_idx in masked_token_index_in_response_list:
            continue
        copied_response[tok_idx][tok_idx + 1] = mask_token_id
        copied_conversation[tok_idx][
            len(encoded_context) - 1 + tok_idx
        ] = mask_token_id

    # Shape of [# of response_token, # of response_token +2, vocab_size]
    model_output = model(copied_response.to(device), return_dict=True).logits
    model_output.detach().cpu().numpy()
    response_logit = softmax(model_output)

    # conversation_logit = (
    #    softmax(model(copied_conversation.to(device), return_dict=True).logits)
    #    .detach()
    #    .cpu()
    #    .numpy()
    # )

    score_list = []

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
