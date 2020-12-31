import json
import logging
import os
import random
from typing import Dict, List
import logging
import numpy as np
import torch
from transformers import (
    BertConfig,
    BertTokenizer,
    EncoderDecoderConfig,
    EncoderDecoderModel,
)


def dump_config(config, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w") as f:
        json.dump(vars(config), f)


def get_logger() -> logging.Logger:
    """Return the Logger class"""
    # create logger
    logger = logging.getLogger("MyLogger")
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    logger.info("Logger Generated")
    return logger


def load_processed_dataset(fname):
    with open(fname, "r") as f:
        ls = [json.loads(el) for el in f.readlines()]
    return ls


def get_span_indices(length, max_span_length=5):
    spans = []
    for span_length in range(max_span_length):
        span_length += 1
        possibile_span_num = length - span_length + 1
        if possibile_span_num <= 0:
            break
        for idx in range(possibile_span_num):
            spans.append([idx, idx + span_length])
    return spans


def set_random_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_encdec_scratch():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Initializing a BERT bert-base-uncased style configuration
    config = BertConfig()
    config.num_hidden_layers = 6
    config.intermediate_size = 2048
    config.hidden_size = 512
    config.num_attention_heads = 8
    config = EncoderDecoderConfig.from_encoder_decoder_configs(config, config)

    # Initializing a Bert2Bert model from the bert-base-uncased style configurations
    model = EncoderDecoderModel(config=config)
    # Accessing the model configuration
    config_encoder = model.config.encoder
    config_decoder = model.config.decoder
    # set decoder config to causal lm
    config_decoder.is_decoder = True
    config_decoder.add_cross_attention = True
    return tokenizer, model


def get_raw_dataset(fname):
    with open(fname, "r") as f:
        ls = [
            json.loads(el.strip())
            for el in f.readlines()
            if "score_list" in el and "-100" not in el
        ]

    return ls


def get_processed_dataset(fname):
    with open(fname, "r") as f:
        ls = [json.loads(el.strip()) for el in f.readlines()]

    return ls


def get_mask_token_index(reply_len):
    num_mask = int(np.random.randint(1, reply_len + 1, 1)[0])
    ids = [_ for _ in range(reply_len)]
    random.shuffle(ids)
    return sorted(ids[:num_mask])


def match_retrieved_response(original_dataset, match_map_fname, db_dataset):
    with open(match_map_fname, "r") as f:
        match_map = [json.loads(el.strip()) for el in f.readlines()]

    assert len(original_dataset) == len(match_map)
    for idx, line in enumerate(original_dataset):
        retrived = match_map[idx]["qq"]  # [0]]["reply"]
        save = [el for el in retrived if el != idx][0]
        retrived = [
            el
            for retrieved_id, el in enumerate(retrived)
            if abs(el - idx) > 3
        ]
        if len(retrived) == 0:
            retrived = save
        else:
            retrived = retrived[0]
        if idx == retrived:
            raise ValueError
        retrived = db_dataset[retrived]["reply"]
        original_dataset[idx]["retrieved"] = retrived
    return original_dataset