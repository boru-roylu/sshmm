import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from data2 import read_data, filter_low_freq_clusters, filter_cluster_seq_lens
from sshmm_utils import StateSplitingHMM

pd.options.display.max_colwidth = 150

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    required=True,
    type=str,
    help="trained model path",
)
parser.add_argument(
    "--exp_dir",
    required=True,
    type=str,
    help="experiment directory that contains json files",
)
parser.add_argument(
    "--output_dir",
    required=True,
    type=str,
    help="output file path",
)
parser.add_argument(
    '--seq_data_parent_dir',
    required=True,
    type=str,
    help='sequence data created by create_seq.py',
)
parser.add_argument(
    '--party',
    required=True,
    type=str,
    help='agent or customer',
)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

splits = ['dev', 'test']

path = os.path.join(args.exp_dir, 'vocab.json')
with open(path, 'r') as f:
    vocab = json.load(f)
    vocab = dict(map(lambda x: (int(x[0]), x[1]), vocab.items()))

path = os.path.join(args.exp_dir, 'info.json')
with open(path, 'r') as f:
    info = json.load(f)
    min_seq_len = info['min_seq_len']
    max_seq_len = info['max_seq_len']

data, _, _ = read_data(args.seq_data_parent_dir, args.party, splits=splits)

ori_avg_len = {split: np.mean(data[split]['x_lens']) for split in splits}
cut_lens = filter_low_freq_clusters(data, vocab)
for split in splits:
    print(f'*************** {split} **************')
    avg_len = np.mean(data[split]['x_lens'])
    avg_cut_len = np.mean(cut_lens[split])
    print(f'    before filter low-freq clusters)')
    print(f'    avg len = {ori_avg_len[split]}')
    print(f'    after filter low-freq clusters')
    print(f'    avg len = {avg_len}')
    print(f'    avg cut len = {avg_cut_len}')
    print(f'    cut percent = {avg_cut_len / ori_avg_len[split]}')

filter_cluster_seq_lens(data, min_seq_len, max_seq_len)
for split in splits:
    print(f'    after filter seq lens # {split} examples = ', len(data[split]['xs']))

topk_clusters = list(vocab.keys())
vocab = {v: k for k, v in vocab.items()}
print('vocab size = ', len(vocab))

model = StateSplitingHMM.load(args.model_path)
#assert set(model.states[0].distribution.parameters[0].keys()) == vocab.values()

print(f'# states = {len(model.states) - 2}')

for split, d in data.items():
    print(f'************** predict {split} ***************')
    rows = []
    for example_id, x in zip(d['example_ids'], d['xs']):
        state_seq = model.predict(x)
        state_seq = ','.join(map(str, state_seq))
        rows.append((example_id, state_seq))

    df = pd.DataFrame(rows, columns=['example_id', 'state_sequence'])
    path = os.path.join(args.output_dir, f'{args.party}_{split}.csv') 
    df.to_csv(path, sep="|", index=False)
