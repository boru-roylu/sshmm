import os
import copy
import types
import numpy as np
import seaborn as sns

from hmmlearn import hmm
from pprint import pprint
from functools import partial
from collections import Counter, OrderedDict, defaultdict

from data import get_datasets
from sshmm import (
    _do_mstep,
    split_state_startprob,
    split_state_transmat,
    split_state_emission,
    entropy,
)

from utils import (
    get_and_plot_ordered_transmat,
    plot_bar,
    save_model,
    get_ordered_idxs,
    get_state_labels,
    get_ordered_emission,
    topk_transmat,
)
    
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--topk_cluster",
    required=True,
    type=int,
    help="topk clusters",
)
parser.add_argument(
    "--targeted_num_states",
    default=20,
    type=int,
    help="Number of total states at the end of training",
)
parser.add_argument(
    "--mixture",
    action="store_true",
    help="Mixture b",
)
parser.add_argument(
    "--topk_trans",
    default=0,
    type=int,
    help="Limit trans to topk",
)
args = parser.parse_args()

np.set_printoptions(precision=2)
sns.set_style('darkgrid')
sns.set(font_scale=0.35)

exp_dir = f"./exp/models_{args.topk_cluster}"
image_dir = f"./images/models_{args.topk_cluster}"

if args.mixture:
    exp_dir += "_mixture"
    image_dir += "_mixture"

if args.topk_trans != 0:
    exp_dir += f"_topk_{args.topk_trans}"
    image_dir += f"_topk_{args.topk_trans}"

os.makedirs(exp_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
os.makedirs(os.path.join(image_dir, "entropy"), exist_ok=True)
os.makedirs(os.path.join(image_dir, "transmat"), exist_ok=True)

"""
    Config
"""
n_iter = 10
random_state=42
num_init_states = 3
init_first_state_prob = 1.0

train_dataset, dev_dataset, vocab, cnt = get_datasets("./data/kmedoids_agent_150", args.topk_cluster)

print('vocab size = ', len(vocab))

num_states = num_init_states
# Initial population probability
startprob = np.zeros(num_states)
startprob[0] = init_first_state_prob
startprob[1:] = (1-init_first_state_prob) / (num_states-1)

transmat = np.array([[0.6, 0.2, 0.2],
                     [0.0, 0.6, 0.4],
                     [0.0, 0.0, 1.0],])


xs = []
ids = []
for x, _id in train_dataset:
    xs.append(x)
    ids.append(_id)
x_lens = [len(x) for x in xs]

# init emission probs
segment_cnts = [Counter(), Counter(), Counter()]
for x in xs:
    segments = np.array_split(x, num_states) 
    for i, c in enumerate(segments):
        segment_cnts[i] += Counter(c.tolist())


single_state_emissionprob = [1] * len(cnt)
for k, v in cnt.items():
    k = vocab[k]
    single_state_emissionprob[k] = v
single_state_emissionprob = np.array([single_state_emissionprob])
single_state_emissionprob = single_state_emissionprob / np.sum(single_state_emissionprob, axis=1, keepdims=True)
single_state_entropy = round(entropy(single_state_emissionprob)[0], 2)
print(f"Single state emissionprob entropy = {single_state_entropy:.2f}")

emissionprobs = []
for segment_cnt in segment_cnts:
    emissionprob = [1] * len(cnt)
    for k, v in segment_cnt.items():
        emissionprob[k] = v
    emissionprobs.append(emissionprob)

emissionprobs = np.array(emissionprobs)
emissionprobs = emissionprobs / np.sum(emissionprobs, axis=1, keepdims=True)

xs = np.concatenate(xs).reshape(-1, 1).astype(int)

# Build an HMM instance and set parameters
model = hmm.MultinomialHMM(n_components=num_states, init_params="",
                           n_iter=n_iter, verbose=True)

model.startprob_ = startprob
model.transmat_ = transmat
model.emissionprob_ = emissionprobs
model._do_mstep = _do_mstep.__get__(model, _do_mstep)

state_idx2parent = OrderedDict([(i, None) for i in range(num_states)])
state_transmat_info = defaultdict(list) #{i: [i] for i in range(num_states)}

old_model = None
split_idx = None
for curr_iter in range(args.targeted_num_states-num_init_states+1):
    print(f"***** current_iter = {curr_iter}; num_states = {num_states} *****")

    print(f"    Split index = {split_idx}")
    ordered_idxs = get_ordered_idxs(state_transmat_info, num_init_states)
    print("    State transmat info = ", state_transmat_info)
    print("    Ordered idxs = ", ordered_idxs)
    emission_entropy = entropy(model.emissionprob_)
    state_labels = get_state_labels(state_idx2parent, ordered_idxs)
    ordered_emission_entropy = get_ordered_emission(emission_entropy, ordered_idxs)
    plot_bar(state_labels, ordered_emission_entropy, os.path.join(image_dir, f"entropy/{num_states}_before_training.eps"), single_state_entropy)
    print(f"    Before training, entropy = {ordered_emission_entropy}")
    print(f"    Before training, average entropy = {np.mean(emission_entropy):.2f}")

    ordered_transmat = get_and_plot_ordered_transmat(model.transmat_, ordered_idxs, os.path.join(image_dir, f"transmat/{num_states}_before_training.eps"), state_labels)
    print("    Before training, transmat = ")
    print(model.transmat_)
    print("    Before training, ordered transmat = ")
    print(ordered_transmat)

    # Training
    model = model.fit(xs, x_lens)
    
    emission_entropy = entropy(model.emissionprob_)
    ordered_emission_entropy = get_ordered_emission(emission_entropy, ordered_idxs) 
    plot_bar(state_labels, ordered_emission_entropy, os.path.join(image_dir, f"entropy/{num_states}_after_training.eps"), single_state_entropy)
    print(f"    After training, entropy = {ordered_emission_entropy}")
    print(f"    After training, average entropy = {np.mean(emission_entropy):.2f}")
    ordered_transmat = get_and_plot_ordered_transmat(model.transmat_, ordered_idxs, os.path.join(image_dir, f"transmat/{num_states}_after_training.eps"), state_labels)
    print("    After training, transmat = ")
    print(model.transmat_)
    print("    After training, ordered transmat = ")
    print(ordered_transmat)
    print("    State_transmat_info = ", end="")
    print(state_transmat_info)

    save_model(num_states, model, exp_dir)

    if old_model:
        print(f"    Last model Log Prob = {old_model.monitor_.history[-1]}")
        print(f"    Last model PPL = {old_model.monitor_.history[-1]/np.mean(x_lens)}")
    print(f"    Current model Log Prob = {model.monitor_.history[-1]}")
    print(f"    Current model PPL = {model.monitor_.history[-1]/np.mean(x_lens)}")
    
    old_model = copy.deepcopy(model)

    # state splitting
    new_state_idx = num_states
    split_idx = np.argmax(emission_entropy)
    state_idx2parent[new_state_idx] = split_idx

    state_transmat_info[split_idx].insert(0, new_state_idx)

    ordered_idxs = get_ordered_idxs(state_transmat_info, num_init_states)

    startprob, _ = split_state_startprob(model.startprob_, split_idx)
    transmat, transmat_mask = split_state_transmat(model.transmat_, split_idx)
    emissionprob, _ = split_state_emission(model.emissionprob_, split_idx, ordered_idxs, mixture=args.mixture)
    num_states += 1

    # copy old model and initialize a new model for next iteration
    old_model = copy.deepcopy(model)
    model = hmm.MultinomialHMM(n_components=num_states, init_params="",
                               n_iter=n_iter, verbose=True)

    _do_mstep = partial(_do_mstep, t_mask=transmat_mask)
    funcType = types.MethodType
    model._do_mstep = funcType(_do_mstep, model) 

    model.startprob_ = startprob
    if args.topk_trans != 0:
        model.transmat_ = topk_transmat(transmat, args.topk_trans)
    else:
        model.transmat_ = transmat
    model.emissionprob_ = emissionprob
    print("\n\n")
