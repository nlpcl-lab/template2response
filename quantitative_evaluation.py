import json
from nlgeval import NLGEval

fname = './generated/dd_generated.jsonl'
with open(fname,'r') as f:
    ls = [json.loads(el) for el in f.readlines()]

nlgeval = NLGEval()
