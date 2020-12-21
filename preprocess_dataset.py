import json
import os

default_path = "./data/raw/"
dailydialog_path = default_path + "ijcnlp_dailydialog/{}/dialogues_{}.txt"
persona_path = default_path + "personachat/{}_both_original.txt"

processed_path = "./data/processed/"
processed_dd_path = processed_path + "dd/"
processed_persona_path = processed_path + "persona/"

pt_data_path = "./data/pt/"
pt_dd_path = pt_data_path + "dd/"
pt_persona_path = pt_data_path + "persona/"


def process_dd():
    setlist = ["train", "validation", "test"]
    for setname in setlist:
        raw_fname = dailydialog_path.format(setname, setname)
        processed_fname = processed_dd_path + f"{setname}.jsonl"
        pt_fname = pt_dd_path + f"{setname}.txt"

        assert os.path.exists(raw_fname)
        with open(raw_fname, "r") as f:
            ls = [
                [uttr.strip() for uttr in el.strip().split("__eou__") if len(uttr) != 0]
                for el in f.readlines()
            ]
        processed_data = []

        for_pt_data = []
        for conversation in ls:
            for turn_index in range(len(conversation) - 1):
                context = conversation[: turn_index + 1]
                reply = conversation[turn_index + 1]
                processed_data.append({"context": context, "reply": reply})
            for_pt_data.append(" ".join(conversation))

        with open(processed_fname, "w") as f:
            for item in processed_data:
                json.dump(item, f)
                f.write("\n")

        with open(pt_fname, "w") as f:
            f.write("\n".join(for_pt_data))


def process_persona():
    setlist = ["train", "valid", "test"]
    for setname in setlist:
        raw_fname = persona_path.format(setname)
        assert os.path.exists(raw_fname)
        processed_fname = processed_persona_path + f"{setname}.jsonl"
        pt_fname = pt_persona_path + f"{setname}.txt"

        with open(raw_fname, "r") as f:
            ls = [el.strip() for el in f.readlines()]
        data = []
        my_per = []
        your_per = []
        conv = []
        for_pt_data = []

        for line in ls:
            if "your persona:" in line:
                if conv != []:
                    for turn_index in range(len(conv) - 1):
                        context = conv[: turn_index + 1]
                        reply = conv[turn_index + 1]
                        data.append(
                            {
                                "your_persona": my_per,
                                "parter_persona": your_per,
                                "context": context,
                                "reply": reply,
                            }
                        )
                    for_pt_data.append(" ".join(conv))
                    my_per, your_per, conv = [], [], []

                line = line.split("your persona:")[-1].strip()
                my_per.append(line)
            elif "s persona:" in line:
                your_per.append(line.split("s persona:")[-1].strip())
            else:
                line = line.split("\t")
                line[0] = " ".join(line[0].split()[1:])
                assert len(line) == 4
                conv.extend(line[:2])

        with open(processed_fname, "w") as f:
            for line in data:
                json.dump(line, f)
                f.write("\n")

        with open(pt_fname, "w") as f:
            f.write("\n".join(for_pt_data))


if __name__ == "__main__":
    os.makedirs(processed_dd_path, exist_ok=True)
    os.makedirs(processed_persona_path, exist_ok=True)
    os.makedirs(pt_dd_path, exist_ok=True)
    os.makedirs(pt_persona_path, exist_ok=True)

    process_dd()
    process_persona()
