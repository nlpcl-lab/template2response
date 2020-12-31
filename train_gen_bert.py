"""
Mask-Predict으로 생성을 하는 BERT를 학습해 boza
"""
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
from transformers import BertForMaskedLM, BertTokenizer


from utils import (
    get_encdec_scratch,
    get_processed_dataset,
    get_mask_token_index,
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
parser.add_argument(
    "--model",
    type=str,
    default="MaskResBert",
    choices=["MaskResBert"],
)

parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--total_epoch", type=int, default=3)
parser.add_argument("--max_seq_len", type=int, default=512)


args = parser.parse_args()

# https://github.com/ddehun/ArgumentSimilarityDomainAdaptation/blob/master/adaptation/trainer.py#L389
class MaskDataset(Dataset):
    def __init__(self, raw_dataset, tokenizer, max_seq_len, is_persona):
        self.is_persona = is_persona
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.feature = self.featurize(raw_dataset)

    def __len__(self):
        return len(self.feature[0])

    def __getitem__(self, idx):
        return tuple([el[idx] for el in self.feature])

    def featurize(self, dataset):
        input_ids_list, input_mask_list, label_ids_list = [
            [] for _ in range(3)
        ]

        for idx, item in enumerate(dataset):
            if idx % 100 == 0:
                print(f"{idx}/{len(dataset)}")

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
            full_conversation = context + reply
            tokenized_context = self.tokenizer.tokenize(context)
            tokenized_reply = self.tokenizer.tokenize(reply)
            reply_length = len(tokenized_reply)
            tokenized_conversation = self.tokenizer.tokenize(
                full_conversation
            )

            assert len(tokenized_context) + len(tokenized_reply) == len(
                tokenized_conversation
            )

            encoded_conversation = self.tokenizer(
                full_conversation,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            try:
                assert (
                    int(sum(encoded_conversation["attention_mask"][0]))
                    == len(tokenized_conversation) + 2
                )
            except:
                print("LENGTH?")
                assert len(tokenized_conversation) + 2 > 512
                continue

            input_mask_list.append(encoded_conversation["attention_mask"][0])
            input_ids = list(
                encoded_conversation["input_ids"][0].detach().clone().numpy()
            )
            mask_ids = get_mask_token_index(reply_length)
            input_ids = [
                el
                if idx - len(tokenized_context) - 1 not in mask_ids
                else self.tokenizer.mask_token_id
                for idx, el in enumerate(input_ids)
            ]

            input_ids_list.append(torch.tensor(input_ids))

            label = [
                el
                if len(tokenized_context) + 1
                <= idx
                < len(tokenized_conversation) + 2
                and idx - len(tokenized_context) - 1 in mask_ids
                else -100
                for idx, el in enumerate(
                    list(
                        encoded_conversation["input_ids"][0]
                        .detach()
                        .clone()
                        .numpy()
                    )
                )
            ]
            label_ids_list.append(torch.tensor(label))

            assert (
                len(input_ids_list)
                == len(input_mask_list)
                == len(label_ids_list)
            )

        return (
            torch.stack(input_ids_list),
            torch.stack(input_mask_list),
            torch.stack(label_ids_list),
        )


def main():
    logger = get_logger()
    device = torch.device("cuda")
    set_random_seed()
    model_name = args.model

    """
    Path definition
    """
    exp_path = "./logs/{}-{}/".format(model_name, args.dataset)
    dump_config(args, exp_path + "config.json")
    model_path, board_path = (
        exp_path + "model/",
        exp_path + "board/",
    )

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(board_path, exist_ok=True)
    writer = SummaryWriter(board_path)

    """
    Load Model
    """
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")

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
    logger.info("Train: {}\nValid:{}".format(len(train_raw), len(valid_raw)))

    train_dataset = MaskDataset(
        train_raw,
        tokenizer,
        args.max_seq_len,
        args.dataset == "persona",
    )
    valid_dataset = MaskDataset(
        valid_raw,
        tokenizer,
        args.max_seq_len,
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
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )
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

    criteria = CrossEntropyLoss(ignore_index=-100)
    label = torch.tensor([_ for _ in range(args.batch_size)]).to(device)
    for epoch in range(args.total_epoch):

        model.train()
        for step, batch in enumerate(train_dataloader):
            input_ids, input_masks, labels = [el.to(device) for el in batch]
            output = model(
                input_ids=input_ids,
                attention_mask=input_masks,
                return_dict=True,
            )["logits"]

            loss = criteria(
                output.view(-1, len(tokenizer)),
                labels.view(-1),
            )

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
                input_ids, input_masks, labels = [
                    el.to(device) for el in batch
                ]

                output = model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    return_dict=True,
                )["logits"]

                loss = criteria(
                    output.view(-1, len(tokenizer)), labels.view(-1)
                )
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
