import os
import json
import argparse
import numpy as np
from pomegranate import *

from data2 import get_datasets
from utils import get_states, entropy
from sshmm_utils import StateSplitingHMM

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_clusters',
    default=50,
    type=int,
    help='num of topk clusters',
)
parser.add_argument(
    '--num_split',
    default=12,
    type=int,
    help='num of state-splitting',
)
parser.add_argument(
    '--max_iterations',
    default=10,
    type=int,
    help='num of iterations for each EM',
)
parser.add_argument(
    '--plot_topk_clusters',
    default=5,
    type=int,
    help='only plot topk clusters for each state',
)
parser.add_argument(
    '--plot_insert_topk_clusters',
    default=5,
    type=int,
    help='only plot topk clusters for each state',
)
parser.add_argument(
    '--exp_parent_dir',
    required=True,
    type=str,
    help='parent dir to save models and images',
)
parser.add_argument(
    '--seq_data_parent_dir',
    required=True,
    type=str,
    help='sequence data created by create_seq.py',
)
parser.add_argument(
    '--center_path',
    required=True,
    type=str,
    help='path of cluster representatives',
)
parser.add_argument(
    '--topk_outgoing',
    default=5,
    type=int,
    help='prune outgoing paths and only keep topk outgoing paths',
)
parser.add_argument(
    '--skip_insert_clusters',
    nargs='+',
    default=[],
    type=int,
    help='',
)
parser.add_argument(
    '--insert',
    action='store_true',
    help='using insertion state',
)
parser.add_argument(
    '--manual_center',
    action='store_true',
    help='manual label',
)
args = parser.parse_args()

num_init_states = 3
exp_dir = os.path.join(args.exp_parent_dir, f'{args.num_clusters:03}')
image_dir = os.path.join(exp_dir, 'images')
model_dir = os.path.join(exp_dir, 'models')
os.makedirs(image_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

train_dataset, dev_dataset, vocab, cnt = get_datasets(args.seq_data_parent_dir, args.num_clusters)
print('Number of top k clusters (vocab size) = ', len(vocab))

"""
    prepare data
"""
xs = []
ids = []
for x, _id in train_dataset:
    xs.append(x)
    ids.append(_id)
xs = xs
ids = ids
x_lens = [len(x) for x in xs]

init_threshold = int(np.mean(x_lens))
sshmm = StateSplitingHMM(args, cnt)
sshmm.init_model(xs, num_init_states, init_threshold=init_threshold)
sshmm.plot(image_path=os.path.join(image_dir, f'sshmm_init'))


print('Start training')
print(f'********** iteration 0 **********')
sshmm.model = StateSplitingHMM.fit(sshmm.model, xs, args.max_iterations)
sshmm.plot(image_path=os.path.join(image_dir, f'sshmm_{sshmm.num_states:02}'))

for iteration in range(args.num_split):
    print(f'*'*20, f'iteration {iteration+1}', '*'*20)
    model_json = json.loads(sshmm.model.to_json())
    state2emissionprob, _, _ = get_states(model_json)
    max_entropy_state = max(state2emissionprob.items(), key=lambda x: entropy(list(x[1].values())))[0]

    print("    Try temperal split")
    t_model, t_logprob, t_new = sshmm.fit_split(xs, max_entropy_state, sshmm.temperal_split)
    print()

    print("    Try vertical split")
    v_model, v_logprob, v_new = sshmm.fit_split(xs, max_entropy_state, sshmm.vertical_split)
    print()

    print(f"    LogProb: temperal = {t_logprob:.3f}; vertical = {v_logprob:.3f}")

    if t_logprob > v_logprob:
        print("    Choose temperal split")
        sshmm.model = t_model
        sshmm.state2child[t_new] = sshmm.state2child[max_entropy_state]
        sshmm.num_temperal_split += 1
    else:
        print("    Choose vertical split")
        sshmm.model = v_model
        sshmm.state2child[v_new] = sshmm.state2child[max_entropy_state]
        sshmm.num_vertical_split += 1
    sshmm.num_states += 1

    print(f'    num_states = {sshmm.num_states}')
    print(f'    num_temperal_split = {sshmm.num_temperal_split}')
    print(f'    num_vertical_split = {sshmm.num_vertical_split}')

    sshmm.plot(image_path=os.path.join(image_dir, f'sshmm_{sshmm.num_states:02}'))
    sshmm.save(model_dir)
