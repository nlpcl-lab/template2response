from typing import Dict, List, Union

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from random import randint, sample
from tqdm import tqdm

import os, pickle


class MaskFillDataset(Dataset):
    def __init__(
        self,
        tokenizer: BertTokenizer,
        raw_dataset: List[Dict[str, Union[List[str], str]]],
        max_sequence_length: int = 512,
        feature_path=None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_sequence_length
        if feature_path is not None and os.path.exists(feature_path):
            with open(feature_path, "rb") as f:
                self.feature = pickle.load(f)
        else:
            featured = self.featurize(raw_dataset)
            if feature_path is not None:
                path = os.path.dirname(feature_path)
                os.makedirs(path, exist_ok=True)
                with open(feature_path, "wb") as f:
                    pickle.dump(featured, f)
            self.feature = featured

    def __getitem__(self, index):
        return tuple([el[index] for el in self.feature])

    def __len__(self):
        return len(self.feature[0])

    def featurize(self, data):
        (
            enc_ids,
            enc_masks,
            dec_input_ids,
            dec_target_ids,
            dec_masks,
        ) = [[] for _ in range(5)]

        for example_index, example in enumerate(tqdm(data)):
            context, response, score_list = (
                example["context"],
                example["reply"],
                example["score_list"],
            )
            assert isinstance(context,str)
            tokenized_context = self.tokenizer(
                context,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            tokenized_response = self.tokenizer(
                response,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            decoder_ids = tokenized_response["input_ids"][0]
            enc_ids.append(tokenized_context["input_ids"][0])
            enc_masks.append(tokenized_context["attention_mask"][0])
            dec_masks.append(tokenized_response["attention_mask"][0])

            num_token = int(sum(dec_masks[-1]).numpy())
            mask_num = randint(0, num_token - 1)
            mask_indices = sample([_ for _ in range(num_token)], mask_num)
            dec_input = [
                self.tokenizer.mask_token_id
                if tok_idx in mask_indices
                else int(tok_id)
                for tok_idx, tok_id in enumerate(decoder_ids)
            ]
            dec_target = [
                int(tok_id)
                if tok_idx not in mask_indices and tok_idx < num_token
                else -100
                for tok_idx, tok_id in enumerate(decoder_ids)
            ]

            dec_input_ids.append(dec_input)
            dec_target_ids.append(dec_target)

        return (
            torch.stack(enc_ids),
            torch.stack(enc_masks),
            torch.tensor(dec_input_ids),
            torch.tensor(dec_target_ids),
            torch.stack(dec_masks),
        )
