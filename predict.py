import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from data import get_datasets

pd.options.display.max_colwidth = 150

parser = argparse.ArgumentParser()
parser.add_argument(
    "--topk_cluster",
    required=True,
    type=int,
    help="topk clusters",
)
parser.add_argument(
    "--model_path",
    required=True,
    type=str,
    help="trained model path",
)
parser.add_argument(
    "--centroid_path",
    required=True,
    type=str,
    help="csv file: cluster id to centroid utterance",
)
parser.add_argument(
    "--raw_path",
    required=True,
    type=str,
    help="csv file: raw clustering file",
)
parser.add_argument(
    "--output_path",
    required=True,
    type=str,
    help="output file path",
)

args = parser.parse_args()

train_dataset, dev_dataset, vocab, cnt = get_datasets("./data/kmedoids_agent_150", args.topk_cluster)
topk_cluster_ids = list(vocab.keys())
vocab = {v: k for k, v in vocab.items()}
print('vocab size = ', len(vocab))

with open(args.model_path, "rb") as f:
    model = pickle.load(f)

df = pd.read_csv(args.centroid_path)
raw_df = pd.read_csv(args.raw_path, compression="gzip")

xs = []
ids = []
for x, _id in dev_dataset:
    xs.append(x)
    ids.append(_id)
x_lens = [len(x) for x in xs]

e = model.emissionprob_
n_states, n_clusters = e.shape
assert n_clusters == args.topk_cluster

dfs = []
for _id, x, x_len in tqdm(zip(ids, xs, x_lens), total=len(xs)):
    states = model.predict(x.reshape(-1, 1), np.array([x_len]))
    if len(set(states)) > 1:
        cluster_seq = []
        for s in states:
            #cluster_seq.append(np.random.choice(np.arange(n_clusters), p=e[s]))
            cluster_seq.append(np.argmax(e[s]))

        utts = []
        chat = raw_df[raw_df["sourcemediaid"] == _id]
        chat = chat[chat["cluster"].isin(topk_cluster_ids)]
        assert len(chat) == len(states)

        ori_utts = chat["phrase"].tolist()
        ori_clusters = chat["cluster"].tolist()

        for s, c, ou, oc in zip(states, cluster_seq, ori_utts, ori_clusters):
            c = vocab[c]
            utt = df[df["cluster"] == c]["phrase"].tolist()[0]
            utts.append((_id, oc, c, s, utt, ou))
        dfs.append(pd.DataFrame(utts, columns=["sourcemediaid", "original_cluster", "sampled_cluster", "state", "phrase", "original_phrase"]))

df = pd.concat(dfs)
df.to_csv(args.output_path, sep="|", index=False)
