import os
import torch
import numpy as np
import pandas as pd
from collections import Counter


def get_datasets(parent_dir, topk):
    lines = []

    vocab = {}
    data = {"train": [], "dev": []}
    for split in ["train", "dev"]:
        path = os.path.join(parent_dir, f"{split}.csv")
        df = pd.read_csv(path, sep="|")

        seqs = df["cluster_sequence"].apply(lambda x: x.split(",")).tolist()
        seqs = [[int(s) for s in seq] for seq in seqs]

        data[split] = seqs

    vocab = Counter([s for seq in data["train"] for s in seq])
    vocab2 = Counter([s for seq in data["dev"] for s in seq])
    vocab = vocab + vocab2
    vocab = {k: i for i, (k, _) in enumerate(vocab.most_common(topk))}

    for split, d in data.items():
        for i, x in enumerate(d):
            d[i] = filter(lambda xx: xx in vocab, x)

    train_dataset = TextDataset(data["train"], vocab)
    dev_dataset = TextDataset(data["dev"], vocab)

    return train_dataset, dev_dataset, vocab


class TextDataset:
    def __init__(self, data, vocab):
        self.data= data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = list(map(self.vocab.get, self.data[idx]))
        return np.array(x)
