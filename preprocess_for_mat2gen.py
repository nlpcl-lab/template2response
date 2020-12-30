import os, json
from collections import Counter
from nltk import word_tokenize

fname = "./data/processed/{}/{}.jsonl"
output_path = "./data/m2g/{}/{}.txt"

for dataset in [
    "dd",
    "persona",
]:
    for setname in [
        "train",
        "test",
        "valid",
    ]:
        print(dataset, setname)
        read_fname = fname.format(dataset, setname)
        out_fname = output_path.format(dataset, setname)
        output_data = []
        os.makedirs(os.path.dirname(out_fname), exist_ok=True)
        vocab_fname = os.path.join(os.path.dirname(out_fname), "vocab.txt")
        if not os.path.exists(vocab_fname):
            counter = Counter()

        with open(read_fname, "r") as f:
            ls = [json.loads(el.strip()) for el in f.readlines()]
        for idx, line in enumerate(ls):
            if idx % 100 == 0:
                print(dataset, setname, idx, len(ls))

            c, r = " ".join(line["context"]), line["reply"]
            if dataset == "persona":
                p1, p2 = " ".join(line["your_persona"]), " ".join(
                    line["parter_persona"]
                )
                c = p1 + p2 + c
            c = [el.lower() for el in word_tokenize(c)]
            r = [el.lower() for el in word_tokenize(r)]
            counter.update(c + r)
            c = " ".join(c)
            r = " ".join(r)
            output_data.append(" | ".join([c, r, c, r]))

        with open(out_fname, "w") as f:
            f.write("\n".join(output_data))
        vocab = [el[0] for el in counter.most_common(35000)]
        with open(vocab_fname, "w") as f:
            f.write("\n".join(vocab))
