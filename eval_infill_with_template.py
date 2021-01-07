import json
import os
import string
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch

score_absolute_threshold = 100


dataset_name = "dd"
fname = "./generated/span4-{}-test_qr.jsonl".format(dataset_name)
model_path = "./logs/gpt2_infill-{}/model/".format(dataset_name)

with open(fname, "r") as f:
    dataset = [json.loads(el) for el in f.readlines()]

device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained(model_path)
model.to(device)
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


for item_idx, item in enumerate(dataset):
    print(item_idx, len(dataset))

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

    def get_template(tokenized_response, span_score):
        # Likely span -> Unlikely span order
        remain_span_score = [
            el for el in span_score if el[2] > score_absolute_threshold
        ]
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
    template = get_template(tokenized_response, span_indices)
    assert original_length == len(template)
    tmp = []
    for idx, el in enumerate(template):
        if idx == len(template) - 1:
            if el == "[MASK]":
                el = BLANK_TOKEN
            tmp.append(el)
        else:
            if el != "[MASK]":
                tmp.append(el)
            else:
                if template[idx + 1] == "[MASK]":
                    continue
                else:
                    tmp.append(BLANK_TOKEN)
    template = tmp[:]

    input_sequence = (
        "[SEPT]".join(context) + "[SEPT]" + " ".join(template) + INFILL_TOKEN
    )
    input_sequence = input_sequence.replace(" " + BLANK_TOKEN, BLANK_TOKEN)

    input_sequence = tokenizer.encode(input_sequence, return_tensors="pt").to(
        device
    )

    model.eos_token_id = tokenizer.eos_token_id
    result = model.generate(
        input_sequence,
        bos_token_id=tokenizer(INFILL_TOKEN)["input_ids"][0],
        eos_token_id=tokenizer.eos_token_id,
        do_sampling=False,
        # num_beams=5,
        max_length=512,
    )

    result = tokenizer.decode(result[0])
    print(result)
    input()
    dataset[item_idx]["generated"] = result


import json

with open(
    "/".join(model_path.split("/")[:-2])
    + "/"
    + "beam_pppl{}.jsonl".format(score_absolute_threshold),
    "w",
) as f:
    for line in dataset:
        json.dump(line, f)
        f.write("\n")
