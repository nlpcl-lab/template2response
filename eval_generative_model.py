import argparse
import json
import os

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
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import MaskFillDataset
from utils import (
    get_encdec_scratch,
    get_processed_dataset,
    set_random_seed,
    get_logger,
    dump_config,
    match_retrieved_response,
    get_mask_token_index,
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
parser.add_argument(
    "--model",
    type=str,
    choices=[
        "gpt2",
        "transformer",
        "bart",
        "rtvNrfn",
        "proto",
        "gpt2_refine",
        "gpt2_infill",
    ],
)
args = parser.parse_args()

BLANK_TOKEN = "[BLANK]"
ANSWER_TOKEN = "[ANSWER]"
INFILL_TOKEN = "[SEP]"
RTV_TOKEN = "[RTV]"


def main():
    print("\n" + "-" * 50)
    logger = get_logger()
    device = torch.device("cuda")
    set_random_seed()
    model_name = args.model

    """
    Path definition
    """
    exp_path = "./logs/{}-{}/".format(model_name, args.dataset)
    dump_config(args, exp_path + "config.json")
    model_path = exp_path + "model/"

    """
    Load Model
    """
    use_enc_dec = model_name in ["transformer"]

    if model_name == "transformer":
        tokenizer, model = get_encdec_scratch()
        model = model.from_pretrained(model_path)
    elif model_name in ["gpt2", "gpt2_refine", "gpt2_infill"]:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        raise NotImplementedError(model_name)

    SPECIAL_TOKENS = [
        "[PERSONA1]",
        "[PERSONA2]",
        "[SEPT]",
    ]

    if model_name == "gpt2_infill":
        SPECIAL_TOKENS.extend([BLANK_TOKEN, ANSWER_TOKEN, INFILL_TOKEN])

    tokenizer.add_tokens(SPECIAL_TOKENS)
    if "refine" in args.model:
        tokenizer.add_tokens([RTV_TOKEN])

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token = "[PAD]"

    if args.model == "transformer":
        model.encoder.resize_token_embeddings(len(tokenizer))
        model.decoder.resize_token_embeddings(len(tokenizer))
    else:
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    """
    Loading dataset
    """
    test_raw = get_processed_dataset(
        args.data_path.format(args.dataset, "test")
    )
    train_raw = get_processed_dataset(
        args.data_path.format(args.dataset, "train")
    )

    test_raw = match_retrieved_response(
        test_raw,
        "./data/repr/{}/test_top_sorted.jsonl".format(args.dataset),
        train_raw,
        "qr",
    )

    model.eval()
    final_output_list = []

    for idx, item in enumerate(test_raw):
        final_output = {}

        context, retrieved, golden = (
            item["context"],
            item["retrieved"],
            item["reply"],
        )
        input_seq = "[SEPT]".join(context) + "[SEPT]"
        if args.dataset == "persona":
            your_persona = (
                "[PERSONA1]".join(item["your_persona"]) + "[PERSONA1]"
            )
            partner_persona = (
                "[PERSONA2]".join(item["parter_persona"]) + "[PERSONA2]"
            )
            input_seq = your_persona + partner_persona + input_seq

        if "refine" in model_name:
            input_seq = retrieved + RTV_TOKEN + input_seq

        final_output["context"] = input_seq
        final_output["golden"] = golden

        tokenized_input = tokenizer(
            input_seq,
            return_tensors="pt",
        )["input_ids"]
        try:
            if model_name == "transformer":
                generated = model.generate(
                    tokenized_input.to(device),
                    bos_token_id=tokenizer.cls_token_id,
                    eos_token_id=tokenizer.sep_token_id,
                    max_length=512,
                    do_sample=False,
                    early_stopping=False,
                    num_beams=5,
                )
            else:
                generated = model.generate(
                    tokenized_input.to(device),
                    max_length=512,
                    do_sample=False,
                    early_stopping=False,
                    num_beams=5,
                )
        except:
            final_output_list.append(final_output)
            continue
        if model_name in ["gpt2", "gpt2_refine"]:
            generated = tokenizer.decode(
                generated[0][len(tokenized_input[0]) :]
            )
        else:
            generated = tokenizer.decode(
                generated[0]
            )  # [len(tokenized_input[0]) :])
        print(generated)
        final_output["final_output"] = generated
        final_output_list.append(final_output)

    os.makedirs("model_generated", exist_ok=True)
    with open(
        "./model_generated/{}_{}_beam5.jsonl".format(
            model_name, args.dataset
        ),
        "w",
    ) as f:
        for line in final_output_list:
            json.dump(line, f)
            f.write("\n")


if __name__ == "__main__":
    main()
