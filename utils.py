import json
import os


def load_processed_dataset(fname):
    with open(fname, "r") as f:
        ls = [json.loads(el) for el in f.readlines()]
    return ls
