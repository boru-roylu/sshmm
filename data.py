import os
import torch
import numpy as np
import pandas as pd
from collections import Counter


def get_datasets(parent_dir, topk):
    lines = []

    vocab = {}
    data = {"train": {}, "dev": {}}
    for split in ["train", "dev"]:
        path = os.path.join(parent_dir, f"{split}.csv")
        df = pd.read_csv(path, sep="|")

        seqs = df["cluster_sequence"].apply(lambda x: x.split(",")).tolist()
        seqs = [[int(s) for s in seq] for seq in seqs]
        ids = df["sourcemediaid"].tolist()

        data[split]["seqs"] = seqs
        data[split]["ids"] = ids

    cnt = Counter([s for seq in data["train"]["seqs"] for s in seq])
    cnt2 = Counter([s for seq in data["dev"]["seqs"] for s in seq])
    cnt = cnt + cnt2
    cnt = dict(cnt.most_common(topk))
    vocab = {k: i for i, k in enumerate(cnt.keys())}

    for split, d in data.items():
        new_seqs = []
        new_ids = []
        for i, (x, _id) in enumerate(zip(d["seqs"], d["ids"])):
            x = list(filter(lambda xx: xx in vocab, x))
            if x:
                new_seqs.append(x)
                new_ids.append(_id)
        data[split]["seqs"] = new_seqs
        data[split]["ids"] = new_ids

    train_dataset = TextDataset(data["train"], vocab)
    dev_dataset = TextDataset(data["dev"], vocab)

    return train_dataset, dev_dataset, vocab, cnt


class TextDataset:
    def __init__(self, data, vocab):
        self.data= data
        self.vocab = vocab

    def __len__(self):
        return len(self.data["seqs"])

    def __getitem__(self, idx):
        x = list(map(self.vocab.get, self.data["seqs"][idx]))
        _ids = self.data["ids"][idx]
        return np.array(x), _ids
