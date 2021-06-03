import os
import torch
import numpy as np
import pandas as pd
from collections import Counter, defaultdict


def pop_list_by_idxs(lst, pop_idxs):
    for i in pop_idxs[::-1]:
        lst.pop(i)
    return lst


def read_data(parent_dir, topk, splits=['train']):
    lines = []

    vocab = {}
    data = {s: {} for s in splits}
    for split, d in data.items():
        path = os.path.join(parent_dir, f"agent_{split}_cluster_sequence.csv")
        df = pd.read_csv(path, sep="|")

        example_ids = df["example_id"].tolist()
        xs = df["cluster_sequence"].apply(
            lambda x: list(map(int, x.split(",")))
        ).tolist()
        x_lens = [len(x) for x in xs]

        d["example_ids"] = example_ids
        d["xs"] = xs
        d["x_lens"] = x_lens
                                          

    cnt = Counter([xx for x in data["train"]["xs"] for xx in x])
    vocab = {
        k: i for i, (k, v) in enumerate(
            sorted(cnt.items(), key=lambda x: x[1], reverse=True)[:topk]
        )
    }

    return data, vocab, cnt


def filter_low_freq_clusters(data, vocab):
    cut_lens = defaultdict(list)
    for split, d in data.items():
        pop_idxs = []
        for i, x in enumerate(d['xs']):
            x = list(filter(lambda xx: xx in vocab, x))
            if not x:
                pop_idxs.append(i)
            else:
                cut_lens[split].append(d['x_lens'][i] - len(x))
                d['xs'][i] = x
                d['x_lens'][i] = len(x)

        d['example_ids'] = pop_list_by_idxs(d['example_ids'], pop_idxs)
        d['xs'] = pop_list_by_idxs(d['xs'], pop_idxs)
        d['x_lens'] = pop_list_by_idxs(d['x_lens'], pop_idxs)

    return cut_lens


def filter_cluster_seq_lens(data, min_seq_len, max_seq_len):
    for split, d in data.items():
        pop_idxs = []
        for i, x_len in enumerate(d['x_lens']):
            if x_len < min_seq_len or x_len > max_seq_len:
                pop_idxs.append(i)

        d['example_ids'] = pop_list_by_idxs(d['example_ids'], pop_idxs)
        d['xs'] = pop_list_by_idxs(d['xs'], pop_idxs)
        d['x_lens'] = pop_list_by_idxs(d['x_lens'], pop_idxs)
