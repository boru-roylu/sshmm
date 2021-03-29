import sys
import copy
import pickle
import types
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from functools import partial
from pprint import pprint

from hmmlearn import hmm
from hmmlearn.utils import normalize


from data import get_datasets
from sshmm import _do_mstep, split_state_startprob, split_state_transmat, split_state_emission, entropy

pd.options.display.max_colwidth = 150
topk_cluster = 30

train_dataset, dev_dataset, vocab, cnt = get_datasets("./data/kmedoids_agent_150", topk_cluster)
vocab = {v: k for k, v in vocab.items()}
print('vocab size = ', len(vocab))

model_path = sys.argv[1]
df_path = sys.argv[2]

with open(model_path, "rb") as f:
    model = pickle.load(f)

df = pd.read_csv(df_path)

xs = list(iter(dev_dataset))
x_lens = [len(x) for x in xs]


sample_len = 25
convs = []
for i in range(10):
    x, s = model.sample(sample_len)
    x = x.reshape(-1)

    #while vocab[x[-1]] not in {424, 541}:
    #    print('resample')
    #    x, s = model.sample(sample_len)
    #    x = x.reshape(-1)

    utts = []
    for xx, ss in zip(x, s):
        xx = vocab[xx]
        utt = df[df["cluster"] == xx]["phrase"].tolist()[0].replace("dot ", " ")
        utts.append((xx, ss, utt))

    sample_df = pd.DataFrame(utts, columns=["cluster", "state", "phrase"])
    convs.append(sample_df)

pprint(convs)
