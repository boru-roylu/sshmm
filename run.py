import os
import copy
import pickle
import types
import numpy as np
from sklearn.utils import check_random_state
from functools import partial
from collections import Counter, OrderedDict
from pprint import pprint
import pandas as pd

from hmmlearn import hmm
from hmmlearn.utils import normalize

from data import get_datasets
from sshmm import _do_mstep, split_state_startprob, split_state_transmat, split_state_emission, entropy

import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(precision=3)
sns.set_style('darkgrid')
sns.set(font_scale=0.5)

topk_cluster = 30
exp_dir = f"./exp/models_{topk_cluster}"
image_dir = f"./images/models_{topk_cluster}"
os.makedirs(exp_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
os.makedirs(os.path.join(image_dir, "entropy"), exist_ok=True)
os.makedirs(os.path.join(image_dir, "transmat"), exist_ok=True)

n_iter = 10
random_state=42
num_init_states = 3
init_first_state_prob = 1.0
targeted_num_states = 10

def get_ordered_transmat(transmat, state_transmat_info, path):
    ordered_idxs = []
    for i in range(len(state_transmat_info)):
        ordered_idxs.append(i)
        for j in state_transmat_info[i]:
            ordered_idxs.append(j)

    ordered_transmat = transmat[ordered_idxs]
    num_states = len(ordered_transmat)

    df_list = []
    for i in range(num_states):
        for j in range(num_states):
            df_list.append((i, j, ordered_transmat[i][j]))
    df = pd.DataFrame(df_list, columns=["i", "j", "transprob"])

    df = pd.pivot_table(data=df, index='i', values='transprob', columns='j')

    plt.clf()
    sns.heatmap(df, cmap="Reds", annot=True, fmt=".3f", vmin=0, vmax=1)
    plt.savefig(path)


def plot_bar(x, y, path):
    assert len(x) == len(y)
    plt.clf()
    plt.rcParams["axes.labelsize"] = 8
    sns.barplot(x="state_idx", y="entropy", data=pd.DataFrame({"state_idx": x, "entropy": y}))
    plt.savefig(path)


def save_model(num_states, model, exp_dir):
    with open(os.path.join(exp_dir, f"{num_states}.pkl"), "wb") as f:
        model._do_mstep = None
        pickle.dump(model, f)


def get_emission_x_labels(state_idx2parent):
    ret = []
    for k, v in state_idx2parent.items():
        if v:
            ret.append(f"{k}_{v}")
        else:
            ret.append(str(k))
    return ret

train_dataset, dev_dataset, vocab, cnt = get_datasets("./data/kmedoids_agent_150", topk_cluster)

print('vocab size = ', len(vocab))

num_states = num_init_states
# Initial population probability
startprob = np.zeros(num_states)
startprob[0] = init_first_state_prob
startprob[1:] = (1-init_first_state_prob) / (num_states-1)

transmat = np.array([[0.8, 0.1, 0.1],
                     [0.0, 0.8, 0.2],
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
state_transmat_info = {i: [] for i in range(num_states)}

old_model = None
for curr_iter in range(targeted_num_states):
    print(f"***** current_iter = {curr_iter}; num_states = {num_states} *****")

    state_emission_x_labels = get_emission_x_labels(state_idx2parent)

    emission_entropy = entropy(model.emissionprob_)
    plot_bar(state_emission_x_labels, emission_entropy, os.path.join(image_dir, f"entropy/{num_states}_before_training.eps"))
    print(f"    Before training, entropy = {emission_entropy}")
    print(f"    Before training, average entropy = {np.mean(emission_entropy):.2f}")

    get_ordered_transmat(model.transmat_, state_transmat_info, os.path.join(image_dir, f"transmat/{num_states}_before_training.eps"))

    # Training
    model = model.fit(xs, x_lens)
    
    emission_entropy = entropy(model.emissionprob_)
    plot_bar(state_emission_x_labels, emission_entropy, os.path.join(image_dir, f"entropy/{num_states}_after_training.eps"))
    print(f"    After training, entropy = {emission_entropy}")
    print(f"    After training, average entropy = {np.mean(emission_entropy):.2f}")
    get_ordered_transmat(model.transmat_, state_transmat_info, os.path.join(image_dir, f"transmat/{num_states}_after_training.eps"))
    print(model.transmat_)

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

    # search the ancester of the new state
    # ancester will be one of the 
    ancester = state_idx2parent[new_state_idx]
    while state_idx2parent[ancester] is not None:
        ancester = state_idx2parent[ancester]
    state_transmat_info[ancester].append(new_state_idx)

    print("    State_transmat_info = ", end="")
    pprint(state_transmat_info)

    startprob, _ = split_state_startprob(model.startprob_, split_idx)
    transmat, transmat_mask = split_state_transmat(model.transmat_, split_idx)
    emissionprob, _ = split_state_emission(model.emissionprob_, split_idx)
    num_states += 1

    # copy old model and initialize a new model for next iteration
    old_model = copy.deepcopy(model)
    model = hmm.MultinomialHMM(n_components=num_states, init_params="",
                               n_iter=n_iter, verbose=True)

    _do_mstep = partial(_do_mstep, t_mask=transmat_mask)
    funcType = types.MethodType
    model._do_mstep = funcType(_do_mstep, model) 

    model.startprob_ = startprob
    model.transmat_ = transmat
    model.emissionprob_ = emissionprob
    print("\n\n")
