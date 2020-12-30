import argparse
import json
import os

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup

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
parser.add_argument("--threshold", type=float, default=0)
parser.add_argument(
    "--sequential", type=str, default="1shot", choices=["1shot", "sequential"]
)

parser.add_argument("--exp_name", type=str, default="scratch")

parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--total_epoch", type=int, default=3)


args = parser.parse_args()


def main():
    logger = get_logger()
    device = torch.device("cuda")
    set_random_seed()
    tokenizer, model = get_encdec_scratch()
    model = torch.nn.DataParallel(model)
    model.to(device)
    """
    Loading dataset
    """
    train_raw, valid_raw = (
        get_raw_dataset(args.data_path.format("train")),
        get_raw_dataset(args.data_path.format("valid")),
    )
    logger.info("Train: {}\nValid:{}".format(len(train_raw), len(valid_raw)))

    train_dataset = MaskFillDataset(
        tokenizer,
        train_raw,
        feature_path="./data/feature/train_mask_fill.pck",
    )
    valid_dataset = MaskFillDataset(
        tokenizer,
        valid_raw,
        feature_path="./data/feature/valid_mask_fill.pck",
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
    Path definition
    """
    exp_path = "./logs/{}/".format(args.exp_name)
    model_path, board_path = exp_path + "model/", exp_path + "board/"
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(board_path, exist_ok=True)
    writer = SummaryWriter(board_path)

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
    for epoch in range(args.total_epoch):
        model.train()
        for step, batch in enumerate(train_dataloader):
            enc_ids, enc_mask, dec_input_ids, dec_target_ids, dec_mask = [
                el.to(device) for el in batch
            ]
            loss = model(
                enc_ids,
                enc_mask,
                dec_input_ids,
                dec_mask,
                labels=dec_target_ids,
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
                enc_ids, enc_mask, dec_input_ids, dec_target_ids, dec_mask = [
                    el.to(device) for el in batch
                ]
                loss = model(
                    enc_ids,
                    enc_mask,
                    dec_input_ids,
                    dec_mask,
                    labels=dec_target_ids,
                    return_dict=True,
                )["loss"]
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
