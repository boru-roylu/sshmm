import os
import json
import argparse
import numpy as np
from pomegranate import *

from data2 import read_data, filter_cluster_seq_lens, filter_low_freq_clusters
from utils import get_states, entropy
from sshmm_utils import StateSplitingHMM

parser = argparse.ArgumentParser()
parser.add_argument(
    '--n_jobs',
    default=1,
    type=int,
    help='num of cpu cores',
)
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
    default=None,
    type=int,
    help='prune outgoing paths and only keep top k outgoing paths',
)
parser.add_argument(
    '--skip_insert_clusters',
    nargs='+',
    default=[],
    type=int,
    help='skip clusters of global emission probs of insertion states',
)
parser.add_argument(
    '--insert',
    action='store_true',
    help='use insertion states',
)
parser.add_argument(
    '--manual_center',
    action='store_true',
    help='use manually labeled centers',
)
parser.add_argument(
    '--self_loop_prob',
    default=0.8,
    type=float,
    help='self loop prob of normal states',
)
parser.add_argument(
    '--insert_self_loop_epsilon',
    default=0.03,
    type=float,
    help='self loop prob of insertion states',
)
parser.add_argument(
    '--max_seq_len_percent',
    default=0.9,
    type=float,
    help='max seq len'
)
parser.add_argument(
    '--min_seq_len_percent',
    default=0.1,
    type=float,
    help='min seq len'
)
parser.add_argument(
    '--party',
    required=True,
    type=str,
    help='agent or customer',
)
args = parser.parse_args()

num_init_states = 3
exp_dir = os.path.join(args.exp_parent_dir, f'{args.party}_{args.num_clusters:03}')
image_dir = os.path.join(exp_dir, 'images')
model_dir = os.path.join(exp_dir, 'models')
os.makedirs(image_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

data, vocab, cnt = read_data(args.seq_data_parent_dir, args.party, args.num_clusters)
print(type(list(vocab.keys())[0]))
print('Number of top k clusters (vocab size) = ', len(vocab))

"""
    prepare data
"""

print('********** Before filter low-freq clusters **********')

ori_avg_len = np.mean(data['train']['x_lens'])
print(f'    avg len = {ori_avg_len}')
print(f"    # train examples = {len(data['train']['xs'])}")

print(data['train']['xs'][0])
cut_lens = filter_low_freq_clusters(data, vocab)
print(data['train']['xs'][0])
exit()

avg_len = np.mean(data['train']['x_lens'])
avg_cut_len = np.mean(cut_lens['train'])

print('********** After filter low-freq clusters **********')
print(f'    avg len = {avg_len}')
print(f'    avg cut len = {avg_cut_len}')
print(f'    cut percent = {avg_cut_len / ori_avg_len}')
print(f"    # train examples = {len(data['train']['xs'])}")


min_seq_len = np.quantile(data['train']['x_lens'], args.min_seq_len_percent)
max_seq_len = np.quantile(data['train']['x_lens'], args.max_seq_len_percent)
filter_cluster_seq_lens(data, min_seq_len, max_seq_len)


print('********** After filter low-freq clusters **********')
print(f'    max_seq_len = {max_seq_len}')
print(f'    min_seq_len = {min_seq_len}')
print(f"    # train examples = {len(data['train']['xs'])}")


"""
    save basic info
"""
path = os.path.join(exp_dir, 'vocab.json')
with open(path, 'w') as f:
    json.dump(vocab, f, indent=4)

path = os.path.join(exp_dir, 'cnt.json')
with open(path, 'w') as f:
    json.dump(cnt, f, indent=4)

path = os.path.join(exp_dir, 'info.json')
info = {'min_seq_len': min_seq_len, 'max_seq_len': max_seq_len}
with open(path, 'w') as f:
    json.dump(info, f, indent=4)

path = os.path.join(exp_dir, 'args.json')
with open(path, 'w') as f:
    json.dump(vars(args), f, indent=4)


init_threshold = int(np.mean(data['train']['x_lens']))
sshmm = StateSplitingHMM(args, cnt)
sshmm.init_model(data['train']['xs'], num_init_states, init_threshold=init_threshold)
sshmm.plot(
    args,
    image_path=os.path.join(image_dir, f'sshmm_init'),
    model=sshmm.model,
    cluster2utt=sshmm.cluster2utt,
)


print('Start training')
print(f'********** iteration 0 **********')
sshmm.model = StateSplitingHMM.fit(
    sshmm.model, data['train']['xs'],
    args.max_iterations,
    args.n_jobs,
)
sshmm.plot(
    args,
    image_path=os.path.join(image_dir, f'sshmm_{sshmm.num_states:02}'),
    model=sshmm.model,
    cluster2utt=sshmm.cluster2utt,
)

for iteration in range(args.num_split):
    print(f'*'*20, f'iteration {iteration+1}', '*'*20, flush=True)
    model_json = json.loads(sshmm.model.to_json())
    state2emissionprob, _, _ = get_states(model_json)
    max_entropy_state = max(state2emissionprob.items(), key=lambda x: entropy(list(x[1].values())))[0]

    print("    Try temperal split")
    t_model, t_logprob, t_new = sshmm.fit_split(data['train']['xs'], max_entropy_state, sshmm.temperal_split)
    print()

    print("    Try vertical split")
    v_model, v_logprob, v_new = sshmm.fit_split(data['train']['xs'], max_entropy_state, sshmm.vertical_split)
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
    print(f'    num_vertical_split = {sshmm.num_vertical_split}', flush=True)

    sshmm.plot(
        args,
        image_path=os.path.join(image_dir, f'sshmm_{sshmm.num_states:02}'),
        model=sshmm.model,
        cluster2utt=sshmm.cluster2utt

    )
    sshmm.save(model_dir)
