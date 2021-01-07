import json
import os
import string
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch


dataset_name = "dd"
fname = "./generated/span4-{}-test_qr.jsonl".format(dataset_name)
model_path = "./logs/gpt2_infill-{}/model/".format(dataset_name)

with open(fname, "r") as f:
    dataset = [json.loads(el) for el in f.readlines()]

device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# model = AutoModelForCausalLM.from_pretrained(model_path)
# model.to(device)
BLANK_TOKEN = "[BLANK]"
ANSWER_TOKEN = "[ANSWER]"
INFILL_TOKEN = "[SEP]"

SPECIAL_TOKENS = [
    "[PERSONA1]",
    "[PERSONA2]",
    "[SEPT]",
    BLANK_TOKEN,
    ANSWER_TOKEN,
    INFILL_TOKEN,
]
tokenizer.add_tokens(SPECIAL_TOKENS)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token = "[PAD]"


PPL_LIST = [20 + 20 * idx for idx in range(15)]
RATIO_COUNTER = {val: [] for val in PPL_LIST}
for item_idx, item in enumerate(dataset):
    if item_idx == 10:
        break
    print(item_idx, "/", len(dataset))

    if "context" not in item:
        continue
    (
        context,
        golden,
        retrieved,
        span_indices,
        span_score,
        tokenized_response,
    ) = (
        item["context"],
        item["golden"],
        item["retrieved"],
        item["span_indices"],
        item["span_score"],
        item["tokenized_response"],
    )

    assert len(span_indices) == len(span_score)
    span_indices = [el for el in span_indices if el[1] - el[0] <= 3]
    span_score = [el for el in span_score if len(el) <= 3]

    for span_order, span in enumerate(span_indices):
        score = span_score[span_order]
        # pseudo-ppl
        score = np.exp(-sum(np.log(score)) / len(score))

        span_indices[span_order].append(score)
    span_indices.sort(key=lambda x: x[2])

    def get_template(tokenized_response, span_score, threshold: int):
        # Likely span -> Unlikely span order
        remain_span_score = [el for el in span_score if el[2] > threshold]
        mask_list = [False for _ in range(len(tokenized_response))]
        for span in remain_span_score:
            mask_list[span[0] : span[1]] = [
                True for _ in range(span[1] - span[0])
            ]
        return [
            el if not mask_list[tok_idx] else "[MASK]"
            for tok_idx, el in enumerate(tokenized_response)
        ]

    original_length = len(tokenized_response)
    for threshold in PPL_LIST:
        template = get_template(tokenized_response, span_indices, threshold)
        assert original_length == len(template)
        mask_count = template.count("[MASK]")
        RATIO_COUNTER[threshold].append(mask_count / original_length)

for k, v in RATIO_COUNTER.items():
    print(f"PPL{k}: {round(100*sum(v)/len(v),2)}%")
