"""
Reimplementation of "Retrieval-guided Dialogue Response Generation via a Matching-to-Generation Framework"
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
parser.add_argument(
    "--model",
    type=str,
    default="match",
    choices=["match"],
)

parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--total_epoch", type=int, default=3)
parser.add_argument("--max_seq_len", type=int, default=512)


args = parser.parse_args()


class RtvDataset(Dataset):
    def __init__(
        self,
        raw_dataset,
        tokenizer,
        max_seq_len: int,
        is_persona: bool,
    ):
        self.is_persona = is_persona
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.feature = self.featurize(raw_dataset)

    def __len__(self):
        return len(self.feature[0])

    def __getitem__(self, idx):
        return tuple([el[idx] for el in self.feature])

    def featurize(self, dataset):
        context_ids, context_masks, response_ids, response_masks = [
            [] for _ in range(4)
        ]

        for idx, item in enumerate(dataset):
            if idx % 100 == 0:
                print(f"{idx}/{len(dataset)}")
            context, reply = "[SEPT]".join(item["context"]), item["reply"]

            if self.is_persona:
                your_persona = (
                    "[PERSONA1]".join(item["your_persona"]) + "[PERSONA1]"
                )
                partner_persona = (
                    "[PERSONA2]".join(item["parter_persona"]) + "[PERSONA2]"
                )

                context = your_persona + partner_persona + context

            tokenized_context = self.tokenizer(
                context,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            tokenized_reply = self.tokenizer(
                reply,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            context_ids.append(tokenized_context["input_ids"][0])
            context_masks.append(tokenized_context["attention_mask"][0])
            response_ids.append(tokenized_reply["input_ids"][0])
            response_masks.append(tokenized_reply["attention_mask"][0])

        return (
            torch.stack(context_ids),
            torch.stack(context_masks),
            torch.stack(response_ids),
            torch.stack(response_masks),
        )


class Matcher(torch.nn.Module):
    def __init__(self, query_bert, response_bert):
        super(Matcher, self).__init__()
        self.qbert = query_bert
        self.rbert = response_bert
        self.q_linear = torch.nn.Linear(768, 768)
        self.r_linear = torch.nn.Linear(768, 768)
        self.w = torch.nn.Linear(768, 768, bias=False)

    def forward(self, context_ids, context_mask, response_ids, response_mask):
        """all parameter shape: [batch_size, max_seq_len]"""
        context_output = self.qbert(
            context_ids,
            context_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        response_output = self.rbert(
            response_ids,
            response_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        context_embedding, response_embedding = (
            context_output["emb"][:, :1],
            response_output["emb"][:, :1],
        )
        # Hope to be [batch_size, max_seq_len, hidden_size]
        context_output = self.q_linear(context_output * context_mask)
        response_output = self.r_linear(response_output * response_mask)

        context_query, context_kv = (
            context_output[:, :1],
            context_output[:, 1:],
        )
        response_query, response_kv = (
            response_output[:, :1],
            response_output[:, 1:],
        )

        context_score = torch.bmm(
            context_query, torch.transpose(context_kv, 1, 2)
        )
        response_score = torch.bmm(
            response_query, torch.transpose(response_kv, 1, 2)
        )
        context,response = 


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
    model = (BertModel.from_pretrained("bert-base-uncased"),)
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

    ConsEnc.resize_token_embeddings(len(tokenizer))
    ResEnc.resize_token_embeddings(len(tokenizer))

    ConsEnc = torch.nn.DataParallel(ConsEnc)
    ResEnc = torch.nn.DataParallel(ResEnc)
    ConsEnc.to(device)
    ResEnc.to(device)

    """
    Loading dataset
    """
    train_raw, valid_raw = (
        get_processed_dataset(args.data_path.format(args.dataset, "train")),
        get_processed_dataset(args.data_path.format(args.dataset, "valid")),
    )
    logger.info("Train: {}\nValid:{}".format(len(train_raw), len(valid_raw)))

    train_dataset = RtvDataset(
        train_raw,
        tokenizer,
        args.max_seq_len,
        args.dataset == "persona",
    )
    valid_dataset = RtvDataset(
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
        list(ConsEnc.parameters()) + list(ResEnc.parameters()),
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
    ConsEnc.module.save_pretrained(context_enc_path)
    ResEnc.module.save_pretrained(response_enc_path)

    """
    Training GOGO
    """
    criteria = CrossEntropyLoss()
    label = torch.tensor([_ for _ in range(args.batch_size)]).to(device)
    for epoch in range(args.total_epoch):
        ConsEnc.train()
        ResEnc.train()
        for step, batch in enumerate(train_dataloader):
            context_ids, context_mask, rseponse_ids, response_mask = [
                el.to(device) for el in batch
            ]
            context_output = ConsEnc(context_ids, context_mask)[1]
            response_output = ResEnc(rseponse_ids, response_mask)[1]

            prediction = torch.mm(context_output, response_output.T)
            loss = criteria(prediction, label)

            loss.backward()
            clip_grad_norm_(ConsEnc.parameters(), 1.0)
            clip_grad_norm_(ResEnc.parameters(), 1.0)
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

        ConsEnc.eval()
        ResEnc.eval()
        total_loss = []
        logger.info("Validation Begin")
        for batch in valid_dataloader:
            with torch.no_grad():
                context_ids, context_mask, rseponse_ids, response_mask = [
                    el.to(device) for el in batch
                ]
                context_output = ConsEnc(context_ids, context_mask)[1]
                response_output = ResEnc(rseponse_ids, response_mask)[1]
                prediction = torch.mm(context_output, response_output.T)
                loss = criteria(prediction, label)
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

        ConsEnc.module.save_pretrained(context_enc_path)
        ResEnc.module.save_pretrained(response_enc_path)


if __name__ == "__main__":
    main()
