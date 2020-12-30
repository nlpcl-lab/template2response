import json, os, argparse
import numpy as np
from numpy import linalg as LA

parser = argparse.ArgumentParser(
    description="Configuration for template generation"
)
parser.add_argument(
    "--dataset", type=str, default="dd", choices=["dd", "persona"]
)
parser.add_argument(
    "--data_fname",
    type=str,
    default="./data/processed/{}/{}.jsonl",
)
parser.add_argument(
    "--repr_fname", type=str, default="./data/repr/{}/{}_{}_repr.npy"
)  # dataset, setname, ctx/res
args = parser.parse_args()


def main():
    train_data = args.data_fname.format(args.dataset, "train")
    with open(train_data, "r") as f:
        train_data = [json.loads(el.strip()) for el in f.readlines()]

    """Train 셋에서 같은 대화 내의 발화에서는 Retrieval을 안해야 하니깐 미리 계산해두기
    """
    same_conv_map = []
    curr_cov = []
    curr_idx = []
    for idx, line in enumerate(train_data):
        if len(curr_cov) == 0:
            curr_cov.extend(line["context"])
            curr_idx.append(idx)
            continue
        if line["context"][: len(curr_cov)] == curr_cov:
            curr_cov.append(line["context"][-1])
            curr_idx.append(idx)
        else:
            same_conv_map.append(curr_idx)
            curr_cov = line["context"]
            curr_idx = [idx]
    if len(curr_idx) != []:
        same_conv_map.append(curr_idx)

    del curr_idx, curr_cov
    tmp_same_conv_map = {}
    for idx, line in enumerate(same_conv_map):
        for el in line:
            tmp_same_conv_map[el] = idx  # index: conversaion ID
    same_conv_map = tmp_same_conv_map

    train_ctx_repr_data = np.load(
        args.repr_fname.format(args.dataset, "train", "ctx"),
        allow_pickle=True,
    )
    train_ctx_repr_data /= LA.norm(train_ctx_repr_data, 2, 1)[:, np.newaxis]
    train_res_repr_data = np.load(
        args.repr_fname.format(args.dataset, "train", "res"),
        allow_pickle=True,
    )
    train_res_repr_data /= LA.norm(train_res_repr_data, 2, 1)[:, np.newaxis]

    for setname in ["test", "train", "valid"]:
        raw_data = args.data_fname.format(args.dataset, setname)
        with open(raw_data, "r") as f:
            raw_data = [json.loads(line.strip()) for line in f.readlines()]

        ctx_repr_data = np.load(
            args.repr_fname.format(args.dataset, setname, "ctx"),
            allow_pickle=True,
        )
        ctx_repr_data /= LA.norm(ctx_repr_data, 2, 1)[:, np.newaxis]

        res_repr_data = np.load(
            args.repr_fname.format(args.dataset, setname, "res"),
            allow_pickle=True,
        )
        res_repr_data /= LA.norm(res_repr_data, 2, 1)[:, np.newaxis]

        assert len(ctx_repr_data) == len(res_repr_data) == len(raw_data)

        for idx, line in enumerate(raw_data):
            if idx % 100 == 0:
                print(setname, f"{idx}/{len(raw_data)}")

            qq_dot = np.dot(ctx_repr_data[idx], train_ctx_repr_data.T)
            qq_sorted = qq_dot.argsort()[::-1][:5]
            qr_dot = np.dot(ctx_repr_data[idx], train_res_repr_data.T)
            qr_sorted = qr_dot.argsort()[::-1][:5]
            rr_dot = np.dot(res_repr_data[idx], train_res_repr_data.T)
            rr_sorted = rr_dot.argsort()[::-1][:5]
            raw_data[idx]["qq"] = [int(el) for el in qq_sorted]
            raw_data[idx]["qr"] = [int(el) for el in qr_sorted]
            raw_data[idx]["rr"] = [int(el) for el in rr_sorted]
        with open(
            "./data/repr/{}/{}_top_sorted.jsonl".format(
                args.dataset, setname
            ),
            "w",
        ) as f:
            for l in raw_data:
                json.dump(l, f)
                f.write("\n")


if __name__ == "__main__":
    main()