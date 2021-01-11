import json, os
import argparse

from nltk import word_tokenize
import numpy as np


def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["dd", "persona"])

    return parser.parse_args()


def main_preprocess_for_scoring():
    args = parse_config()
    output_fname = "./data/m2g/gen/{}/test_template_input.txt"

    with open(
        "./data/processed/{}/train.jsonl".format(args.dataset), "r"
    ) as f:
        db = [json.loads(el.strip()) for el in f.readlines()]
    repr_test_file = "./data/repr/{}/test_top_sorted.jsonl".format(
        args.dataset
    )
    with open(repr_test_file, "r") as f:
        ls = [json.loads(el.strip()) for el in f.readlines()]

    output = []
    for line in ls:
        line = line
        context = " ".join(line["context"])
        if args.dataset == "persona":
            context = (
                " ".join(line["your_persona"])
                + " "
                + " ".join(line["parter_persona"])
                + context
            )
        context = " ".join([el.lower() for el in word_tokenize(context)])
        reply = " ".join(
            [el.lower() for el in word_tokenize(db[line["qq"][0]]["reply"])]
        )
        output.append("|".join([context, reply, context, reply]))
    with open(output_fname.format(args.dataset), "w") as f:
        f.write("\n".join(output))


def main_after_scoring():
    args = parse_config()

    read_fname = "./data/m2g/gen/{}/scored.txt".format(args.dataset)
    with open(read_fname, "r") as f:
        ls = [el.strip().split("|") for el in f.readlines()]
    output_fname = "./data/m2g/test_input/{}_input.txt".format(args.dataset)

    output = []
    for line_idx, line in enumerate(ls):
        assert len(line) == 4
        context = line[0]
        res, score = line[1].split(), [float(el) for el in line[-1].split()]
        assert len(res) == len(score)
        over_zero = [el for el in score if el > 0]
        if len(over_zero) == 0:

            print(line_idx, "all_zero")
            res = res[np.argsort(score)[-1]]
        else:
            avg = sum(over_zero) / len(over_zero)
            indice = [idx for idx, el in enumerate(score) if el > avg]
            res = " ".join(
                [el for idx, el in enumerate(res) if idx in indice]
            )
        output.append(context + "|" + res)

    with open(output_fname, "w") as f:
        for lin in output:
            f.write(lin)
            f.write("\n")


if __name__ == "__main__":
    main_preprocess_for_scoring()
    main_after_scoring()