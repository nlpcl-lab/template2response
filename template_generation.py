import argparse
import json
import os
from pprint import pprint

import torch
from transformers import BertForMaskedLM, BertTokenizer

from utils import load_processed_dataset

parser = argparse.ArgumentParser(description="Configuration for template generation")
parser.add_argument("--dataset", type=str, default="dd", choices=["dd", "persona"])
parser.add_argument(
    "--lm_type",
    type=str,
    default="bert-ft",
    choices=["bert-ft", "bert-scratch", "roberta-ft", "roberta-scratch"],
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
parser.add_argument("--threshold", type=float, default=0)

args = parser.parse_args()


def score_importance(data, model, tokenizer, threshold, device, contain_persona: False):
    softmax = torch.nn.Softmax(dim=2)
    mask_token_id = tokenizer.mask_token_id

    for sample_index, sample in enumerate(data):
        print(f"{sample_index}/{len(data)}")
        context, reply = " ".join(sample["context"]), sample["reply"]
        conversation = context.strip() + " " + reply.strip()

        if contain_persona:
            my_persona, your_persona = (
                sample["my_persona"],
                sample["your_persona"],
            )
        tokenzied_context = tokenizer.tokenize(context)
        tokenzied_reply = tokenizer.tokenize(reply)
        tokenzied_conversation = tokenizer.tokenize(conversation)
        encoded_context = torch.tensor(tokenizer(context)["input_ids"])
        encoded_reply = torch.tensor(tokenizer(reply)["input_ids"])
        encoded_conversation = torch.tensor(tokenizer(conversation)["input_ids"])
        context_length = len(encoded_context)
        if context_length >= 512:
            break
        assert len(tokenzied_context) + 2 == context_length

        res_score_list, conv_score_list = [], []

        tiled_response = encoded_reply.repeat(len(tokenzied_reply), 1)
        tiled_conversation = encoded_conversation.repeat(len(tokenzied_reply), 1)

        original_token_list = []
        for tok_idx, tok_id in enumerate(tokenzied_reply):
            original_token_list.append(
                int(tiled_response[tok_idx][tok_idx + 1].numpy())
            )
            assert (
                tiled_response[tok_idx][tok_idx + 1]
                == tiled_conversation[tok_idx][context_length - 1 + tok_idx]
            )
            tiled_response[tok_idx][tok_idx + 1] = mask_token_id
            tiled_conversation[tok_idx][context_length - 1 + tok_idx] = mask_token_id

        # Shape of [# of response_token, # of response_token +2, vocab_size]
        with torch.no_grad():
            response_logit = softmax(
                model(tiled_response.to(device), return_dict=True).logits
            )
            conversation_logit = softmax(
                model(tiled_conversation.to(device), return_dict=True).logits
            )
        assert len(original_token_list) == len(tokenzied_reply)
        res_score_list, conv_score_list = [], []
        for dim1 in range(len(tokenzied_reply)):
            row, res_col, conv_col, original_token = (
                dim1,
                dim1 + 1,
                context_length - 1 + dim1,
                original_token_list[dim1],
            )
            resscore = response_logit[row][res_col][original_token]
            convscore = conversation_logit[row][conv_col][original_token]
            res_score_list.append(float(resscore.detach().cpu().numpy()))
            conv_score_list.append(float(convscore.detach().cpu().numpy()))

        assert len(tokenzied_reply) == len(res_score_list) == len(conv_score_list)
        data[sample_index]["tokenized_reply"] = tokenzied_reply
        data[sample_index]["response_score"] = res_score_list
        data[sample_index]["conv_score"] = conv_score_list
    return data


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
        print(idx)
        setname = ["valid", "train", "test"][idx]
        dataset = score_importance(
            dataset,
            model,
            tokenizer,
            args.threshold,
            device,
            contain_persona=True if args.dataset == "persona" else False,
        )
        os.makedirs("./score/{}/".format(args.dataset), exist_ok=True)
        with open("./score/{}/{}.jsonl".format(args.dataset, setname), "w") as f:
            for line in dataset:
                json.dump(line, f)
                f.write("\n")


if __name__ == "__main__":
    main()
