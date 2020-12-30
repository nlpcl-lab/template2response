import argparse
import json
import os
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler, Dataset
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    get_linear_schedule_with_warmup,
    BartTokenizer,
    BartModel,
)
from transformers import BertModel, BertTokenizer

from dataset import MaskFillDataset
from utils import (
    get_encdec_scratch,
    get_processed_dataset,
    set_random_seed,
    get_logger,
    dump_config,
)


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

args = parser.parse_args()


def main():
    device = torch.device("cuda")

    exp_path = "./logs/biencoder-{}/".format(args.dataset)
    ConsEnc, ResEnc = (
        BertModel.from_pretrained(exp_path + "model_ContEnc/"),
        BertModel.from_pretrained(exp_path + "model_ResEnc/"),
    )
    ConsEnc.to(device)
    ResEnc.to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    SPECIAL_TOKENS = [
        "[PERSONA1]",
        "[PERSONA2]",
        "[SEPT]",
    ]
    tokenizer.add_tokens(SPECIAL_TOKENS)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token = "[PAD]"

    for setname in ["test", "valid", "train"]:
        in_fname = args.data_path.format(args.dataset, setname)
        out_fname = (
            args.data_path.format(args.dataset, setname + "_{}_repr")
            .replace("jsonl", "npy")
            .replace("/processed/", "/repr/")
        )
        os.makedirs(os.path.dirname(out_fname), exist_ok=True)
        with open(in_fname, "r") as f:
            ls = [json.loads(line.strip()) for line in f.readlines()]

        context_repr_list, response_repr_list = [], []

        for idx, item in enumerate(ls):
            if idx % 100 == 0:
                print(setname, f"{idx}/{len(ls)}")

            context, reply = "[SEPT]".join(item["context"]), item["reply"]
            if args.dataset == "persona":
                your_persona = (
                    "[PERSONA1]".join(item["your_persona"]) + "[PERSONA1]"
                )
                partner_persona = (
                    "[PERSONA2]".join(item["parter_persona"]) + "[PERSONA2]"
                )
                context = your_persona + partner_persona + context
            tokenized_context = tokenizer(
                context,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            tokenized_reply = tokenizer(
                reply,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            context_ids = tokenized_context["input_ids"]
            context_masks = tokenized_context["attention_mask"]
            response_ids = tokenized_reply["input_ids"]
            response_masks = tokenized_reply["attention_mask"]
            with torch.no_grad():
                context_output = (
                    ConsEnc(context_ids.to(device), context_masks.to(device))[
                        1
                    ][0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                response_output = (
                    ResEnc(
                        response_ids.to(device), response_masks.to(device)
                    )[1][0]
                    .detach()
                    .cpu()
                    .numpy()
                )
            context_repr_list.append(context_output)
            response_repr_list.append(response_output)

        np.array(context_repr_list).dump(out_fname.format("ctx"))
        np.array(response_repr_list).dump(out_fname.format("res"))


if __name__ == "__main__":
    main()