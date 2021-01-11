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

parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--total_epoch", type=int, default=3)
parser.add_argument("--max_seq_len", type=int, default=512)


args = parser.parse_args()

BLANK_TOKEN = "[BLANK]"
ANSWER_TOKEN = "[ANSWER]"
INFILL_TOKEN = "[SEP]"


class GenerativeDataset(Dataset):
    def __init__(
        self,
        raw_dataset,
        tokenizer,
        seperate_decoder: bool,
        max_seq_len: int,
        model_name: str,
        tgt_auto_shift: bool,
        is_persona: bool,
    ):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.seperate_decoder = seperate_decoder
        self.max_seq_len = max_seq_len
        self.is_persona = is_persona
        if self.model_name == "gpt2_infill":
            self.feature = self.featurize_gpt_infill(raw_dataset)
        elif self.seperate_decoder:
            self.feature = self.featurize_enc_dec(raw_dataset, tgt_auto_shift)
        else:
            self.feature = self.featurize(raw_dataset, tgt_auto_shift)

    def __len__(self):
        return len(self.feature[0])

    def __getitem__(self, idx):
        return tuple([el[idx] for el in self.feature])

    def featurize_enc_dec(self, dataset, auto_shift):
        enc_ids, enc_mask, dec_ids, dec_mask, tgt_ids = [[] for _ in range(5)]
        for idx, item in enumerate(dataset):
            if idx % 100 == 0:
                print(f"{idx}/{len(dataset)}")

            context, reply = (
                item["context"],
                item["reply"],
            )
            context = "[SEPT]".join(context)
            if self.is_persona:
                your_persona = (
                    "[PERSONA1]".join(item["your_persona"]) + "[PERSONA1]"
                )
                partner_persona = (
                    "[PERSONA2]".join(item["parter_persona"]) + "[PERSONA2]"
                )
                context = your_persona + partner_persona + context
            context = self.tokenizer(
                context,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            reply = self.tokenizer(
                reply,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            enc_ids.append(context["input_ids"][0])
            enc_mask.append(context["attention_mask"][0])
            dec_ids.append(reply["input_ids"][0])
            dec_mask.append(reply["attention_mask"][0])
            tgt_id = list(reply["input_ids"][0].numpy())
            if not auto_shift:
                tgt_id = tgt_id[1:] + [self.tokenizer.pad_token_id]
            tgt_id = [
                el if el != self.tokenizer.pad_token_id else -100
                for el in tgt_id
            ]
            tgt_ids.append(tgt_id)

        return (
            torch.stack(enc_ids),
            torch.stack(enc_mask),
            torch.stack(dec_ids),
            torch.stack(dec_mask),
            torch.tensor(tgt_ids),
        )

    def featurize_gpt_infill(self, dataset):
        input_ids, input_mask, target_ids = [], [], []
        for item_idx, item in enumerate(dataset):
            if item_idx % 100 == 0:
                print(f"{item_idx}/{len(dataset)}")

            context, reply = (
                "[SEPT]".join(item["context"]) + "[SEPT]",
                item["reply"],
            )
            if self.is_persona:
                your_persona = (
                    "[PERSONA1]".join(item["your_persona"]) + "[PERSONA1]"
                )
                partner_persona = (
                    "[PERSONA2]".join(item["parter_persona"]) + "[PERSONA2]"
                )
                context = your_persona + partner_persona + context
            response_tokens = self.tokenizer.tokenize(reply)
            masked_token_list = get_mask_token_index(len(response_tokens))

            unmerged_masksed_part = [
                BLANK_TOKEN if idx in masked_token_list else el
                for idx, el in enumerate(response_tokens)
            ]

            masksed_part, answer_part = [], []
            for idx, tok in enumerate(unmerged_masksed_part):
                if idx == len(unmerged_masksed_part) - 1:
                    masksed_part.append(tok)
                    break
                if tok != BLANK_TOKEN:
                    masksed_part.append(tok)
                    continue
                if unmerged_masksed_part[idx + 1] != BLANK_TOKEN:
                    masksed_part.append(tok)
            for idx, mask_idx in enumerate(masked_token_list):
                if idx == len(masked_token_list) - 1:  # Last word
                    answer_part.append(response_tokens[mask_idx])
                    answer_part.append(ANSWER_TOKEN)
                    break
                if (
                    masked_token_list[idx + 1] == mask_idx + 1
                ):  # Continous masked words
                    answer_part.append(response_tokens[mask_idx])
                    continue
                else:
                    answer_part.append(response_tokens[mask_idx])
                    answer_part.append(ANSWER_TOKEN)

            response_sequence = masksed_part + [INFILL_TOKEN] + answer_part
            input_sequence = (
                self.tokenizer.tokenize(context)
                + response_sequence
                + [self.tokenizer.eos_token]
            )

            tgt_prefix = []
            if True:  # 이게 logs_tmp에 있음
                tgt_prefix = (
                    [-100 for _ in self.tokenizer.tokenize(context)]
                    + [-100 for _ in masksed_part]
                    + [-100]
                )
                target_seq = answer_part + [self.tokenizer.eos_token]
            else:  # Loss for Every Token, including context
                target_seq = (
                    self.tokenizer.tokenize(context)
                    + response_sequence
                    + [self.tokenizer.eos_token]
                )
            assert (
                input_sequence.count(BLANK_TOKEN)
                == input_sequence.count(ANSWER_TOKEN)
                == target_seq.count(ANSWER_TOKEN)
            )
            if len(input_sequence) >= self.max_seq_len:
                continue
            ids_list = self.tokenizer.convert_tokens_to_ids(input_sequence)
            attention_list = [1 for _ in range(len(ids_list))] + [
                0 for _ in range(self.max_seq_len - len(ids_list))
            ]

            input_seq = ids_list + [
                self.tokenizer.pad_token_id
                for _ in range(self.max_seq_len - len(ids_list))
            ]
            target_seq = (
                tgt_prefix
                + self.tokenizer.convert_tokens_to_ids(target_seq)
                + [-100 for _ in range(self.max_seq_len - len(ids_list))]
            )
            input_ids.append(input_seq)
            input_mask.append(attention_list)
            target_ids.append(target_seq)
        return (
            torch.tensor(input_ids),
            torch.tensor(input_mask),
            torch.tensor(target_ids),
        )

    def featurize(self, dataset, auto_shift):
        input_ids, input_mask, target_ids = [[] for _ in range(3)]
        for idx, item in enumerate(dataset):
            if idx % 100 == 0:
                print(f"{idx}/{len(dataset)}")

            context, reply = item["context"], item["reply"]
            input_seq = "[SEPT]".join(context) + "[SEPT]"
            if self.is_persona:
                your_persona = (
                    "[PERSONA1]".join(item["your_persona"]) + "[PERSONA1]"
                )
                partner_persona = (
                    "[PERSONA2]".join(item["parter_persona"]) + "[PERSONA2]"
                )
                input_seq = your_persona + partner_persona + input_seq

            if "refine" in self.model_name:
                input_seq = item["retrieved"] + "[RTV]" + input_seq

            tokenized = self.tokenizer(
                input_seq + reply + self.tokenizer.eos_token,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            context_length = len(self.tokenizer.tokenize(input_seq))
            input_ids.append(tokenized["input_ids"][0])
            input_mask.append(tokenized["attention_mask"][0])
            tgt_id = list(tokenized["input_ids"][0].numpy())
            if not auto_shift:
                tgt_id = tgt_id[1:] + [self.tokenizer.pad_token_id]
            tgt_id = [
                el
                if el != self.tokenizer.pad_token_id
                and token_idx >= context_length
                else -100
                for token_idx, el in enumerate(tgt_id)
            ]

            target_ids.append(tgt_id)
        return (
            torch.stack(input_ids),
            torch.stack(input_mask),
            torch.tensor(target_ids),
        )


def main():
    logger = get_logger()
    device = torch.device("cuda")
    set_random_seed()
    model_name = args.model

    if model_name == "transformer":
        assert args.learning_rate == 1e-4
    else:
        assert args.learning_rate == 2e-5

    """
    Path definition
    """
    exp_path = "./logs/{}-{}/".format(model_name, args.dataset)
    dump_config(args, exp_path + "config.json")
    model_path, board_path = exp_path + "model/", exp_path + "board/"
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(board_path, exist_ok=True)
    writer = SummaryWriter(board_path)

    """
    Load Model
    """
    use_enc_dec = model_name in ["transformer", "bart"]
    tgt_auto_shift = model_name in ["gpt2", "gpt2_refine", "gpt2_infill"]

    if model_name == "transformer":
        tokenizer, model = get_encdec_scratch()
    elif model_name == "bart":
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-medium")
        model = BartModel.from_pretrained("facebook/bart-medium")
    elif model_name in ["gpt2", "gpt2_refine", "gpt2_infill"]:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
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
        tokenizer.add_tokens(["[RTV]"])

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token = "[PAD]"

    if args.model == "transformer":
        model.encoder.resize_token_embeddings(len(tokenizer))
        model.decoder.resize_token_embeddings(len(tokenizer))
    else:
        model.resize_token_embeddings(len(tokenizer))

    model = torch.nn.DataParallel(model)
    model.to(device)

    """
    Loading dataset
    """
    train_raw, valid_raw = (
        get_processed_dataset(args.data_path.format(args.dataset, "train")),
        get_processed_dataset(args.data_path.format(args.dataset, "valid")),
    )

    if args.model == "gpt2_refine":
        train_raw = match_retrieved_response(
            train_raw,
            "./data/repr/{}/train_top_sorted.jsonl".format(args.dataset),
            train_raw,
            "qr",
        )
        valid_raw = match_retrieved_response(
            valid_raw,
            "./data/repr/{}/valid_top_sorted.jsonl".format(args.dataset),
            train_raw,
            "qr",
        )

    logger.info("Train: {}\nValid:{}".format(len(train_raw), len(valid_raw)))

    train_dataset = GenerativeDataset(
        train_raw,
        tokenizer,
        use_enc_dec,
        args.max_seq_len,
        model_name,
        tgt_auto_shift,
        args.dataset == "persona",
    )
    valid_dataset = GenerativeDataset(
        valid_raw,
        tokenizer,
        use_enc_dec,
        args.max_seq_len,
        model_name,
        tgt_auto_shift,
        args.dataset == "persona",
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.batch_size, drop_last=True
    )

    """
    Prepare training
    """
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    step_per_epoch = int(len(train_dataloader))
    total_steps = step_per_epoch * args.total_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )
    global_step = 0

    model.module.save_pretrained(model_path)

    """
    Training GOGO
    """
    for epoch in range(args.total_epoch):
        model.train()
        for step, batch in enumerate(train_dataloader):
            if use_enc_dec:
                enc_ids, enc_mask, dec_input_ids, dec_mask, dec_target_ids = [
                    el.to(device) for el in batch
                ]
                loss = model(
                    enc_ids,
                    enc_mask,
                    dec_input_ids,
                    labels=dec_target_ids,
                    return_dict=True,
                )["loss"].mean()
            else:
                input_ids, input_mask, target_ids = [
                    el.to(device) for el in batch
                ]
                loss = model(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    labels=target_ids,
                    return_dict=True,
                )["loss"].mean()

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            logger.info(
                f"Epoch {epoch} Step {step}/{len(train_dataloader)} Loss {loss}"
            )
            writer.add_scalars(
                "loss", {"train": loss}, global_step=global_step
            )
            writer.flush()

        model.eval()
        total_loss = []
        logger.info("Validation Begin")
        for batch in valid_dataloader:
            with torch.no_grad():
                if use_enc_dec:
                    (
                        enc_ids,
                        enc_mask,
                        dec_input_ids,
                        dec_mask,
                        dec_target_ids,
                    ) = [el.to(device) for el in batch]
                    loss = model(
                        enc_ids,
                        enc_mask,
                        dec_input_ids,
                        labels=dec_target_ids,
                        return_dict=True,
                    )["loss"].mean()
                else:
                    input_ids, input_mask, target_ids = [
                        el.to(device) for el in batch
                    ]
                    loss = model(
                        input_ids,
                        attention_mask=input_mask,
                        labels=target_ids,
                        return_dict=True,
                    )["loss"].mean()

                total_loss.append(float(loss.mean().detach().cpu().numpy()))

        logger.info(
            "Valid Loss {}".format(
                round(sum(total_loss) / len(total_loss), 2)
            )
        )
        writer.add_scalars(
            "loss",
            {"valid": sum(total_loss) / len(total_loss)},
            global_step=global_step,
        )
        writer.flush()

        model.module.save_pretrained(model_path)


if __name__ == "__main__":
    main()
