import argparse
import json
import os
import numpy as np

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    get_linear_schedule_with_warmup,
    BertTokenizer,
    BertForMaskedLM,
)

from dataset import MaskFillDataset
from utils import (
    get_encdec_scratch,
    get_raw_dataset,
    set_random_seed,
    get_logger,
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
    default="./score/response-1shot-ftbert/dd/{}.jsonl",
)

parser.add_argument(
    "--sequential", type=str, default="1shot", choices=["1shot", "sequential"]
)

parser.add_argument("--exp_name", type=str, default="scratch")

parser.add_argument("--num_iter", type=int, default=10)

parser.add_argument("--threshold", type=float, default=0.3)

args = parser.parse_args()


def main():
    logger = get_logger()
    device = torch.device("cuda")
    set_random_seed()
    # tokenizer, model = get_encdec_scratch()
    # model = model.from_pretrained("./logs/{}/model/".format(args.exp_name))
    model = BertForMaskedLM.from_pretrained("./model/lm/dd_ft/")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model.to(device)
    """
    Loading dataset
    """
    valid_raw = get_raw_dataset(args.data_path.format("valid"))
    softmax = torch.nn.Softmax(dim=1)
    dump_output = []
    error_counter = 0

    for idx, batch in enumerate(valid_raw):
        print("{}/{}".format(idx, len(valid_raw)))
        context, reply, score_list = (
            batch["context"],
            batch["tokenized_reply"],
            batch["score_list"],
        )

        assert len(reply) == len(score_list)
        # score_list = [-100 for _ in range(len(score_list))]

        initial_reply = [
            token
            if score_list[idx] > args.threshold
            else tokenizer.mask_token
            for idx, token in enumerate(reply)
        ]

        mask_index_list = [
            idx
            for idx, token in enumerate(initial_reply)
            if token == tokenizer.mask_token
        ]

        bert_input_seq = tokenizer(
            " ".join(context)
            + " "
            + " ".join(initial_reply).replace(" ##", "").strip(),
            return_tensors="pt",
        )["input_ids"].to(device)
        if len(bert_input_seq[0]) > 512:
            dump_output.append({})
            error_counter += 1
            continue
        step = 0
        print("Context: ", end="")
        print(context)
        print("Original Reply: ", end="")
        print(" ".join(reply) + "\n")
        sequence = []
        while True:
            print("STEP: {}".format(step))
            print(mask_index_list)
            print(
                tokenizer.convert_ids_to_tokens(
                    bert_input_seq[0][-len(initial_reply) - 1 :]
                )
            )
            if len(mask_index_list) == 0:
                break
            with torch.no_grad():
                output = model(bert_input_seq, return_dict=True)["logits"][0][
                    -1 - len(reply) : -1
                ]
            assert len(output) == len(reply) == len(initial_reply)
            output = softmax(output)
            max_output = torch.max(output, 1)
            # Max prob of each timestep
            max_prob = max_output.values.detach().cpu().numpy()
            # Most likely word for each timestep
            max_word = max_output.indices.detach().cpu().numpy()
            max_prob = [
                -100 if idx not in mask_index_list else max_prob[idx]
                for idx in range(len(reply))
            ]

            max_word_index = np.argmax(max_prob)
            selected_word = max_word[max_word_index]
            print(
                "Selected Index: {} Word: {}".format(
                    max_word_index,
                    tokenizer._convert_id_to_token(selected_word),
                )
            )
            sequence.append(
                [
                    int(max_word_index),
                    tokenizer._convert_id_to_token(selected_word),
                ]
            )
            bert_input_seq[0][
                -len(reply) - 1 + max_word_index
            ] = selected_word
            mask_index_list.remove(int(max_word_index))
            step += 1
        new_reply = " ".join(
            tokenizer.convert_ids_to_tokens(
                bert_input_seq[0][-len(reply) - 1 : -1]
            )
        )
        print("\nNew Reply\n" + new_reply)
        try:
            assert tokenizer.mask_token not in new_reply
        except:
            dump_output.append({})
            error_counter += 1
            continue

        dump_output.append(
            {
                "context": context,
                "original_reply": reply,
                "changed_reply": new_reply,
                "sequence": sequence,
                "score_list": score_list,
            }
        )
    print(len(dump_output), error_counter)
    with open(
        "./generated/argmax_thd{}_valid.jsonl".format(args.threshold),
        "w"
        # "./generated/argmax_maskall_valid.jsonl",
        # "w",
    ) as f:
        for line in dump_output:
            json.dump(line, f)
            f.write("\n")


if __name__ == "__main__":
    main()
