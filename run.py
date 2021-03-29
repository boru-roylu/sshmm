import os
import copy
import pickle
import types
import numpy as np
from sklearn.utils import check_random_state
from functools import partial
from collections import Counter

from hmmlearn import hmm
from hmmlearn.utils import normalize

from data import get_datasets
from sshmm import _do_mstep, split_state_startprob, split_state_transmat, split_state_emission, entropy

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

def plot_bar(x, y, path):
    assert len(x) == len(y)
    plt.clf()
    sns.barplot(x, y)
    plt.savefig(path)

topk_cluster = 30
train_dataset, dev_dataset, vocab, cnt = get_datasets("./data/kmedoids_agent_150", topk_cluster)
exp_dir = f"./exp/models_{topk_cluster}_update_part_of_transmat"
os.makedirs(exp_dir, exist_ok=True)

print('vocab size = ', len(vocab))

n_iter = 10
random_state=42
n_components = 3
n_features = len(vocab)
init_first_state_prob = 0.99
max_num_splits = 200

##############################################################
# Prepare parameters for a 4-components HMM
# Initial population probability
startprob = np.zeros(n_components)
startprob[0] = init_first_state_prob
startprob[1:] = (1-init_first_state_prob) / (n_components-1)

# The transition matrix, note that there are no transitions possible
# between component 1 and 3
#transmat = np.array([[0.7, 0.2, 0.0, 0.1],
#                     [0.0, 0.5, 0.2, 0.0],
#                     [0.0, 0.0, 0.5, 0.2],
#                     [0.0, 0.0, 0.0, 0.6]])
transmat = np.array([[0.6, 0.2, 0.2],
                     [0.0, 0.6, 0.4],
                     [0.0, 0.6, 0.4],])


#rs = check_random_state(random_state)
#emissionprob = rs.rand(n_components, n_features)
#normalize(emissionprob, axis=1)

xs = []
ids = []
for x, _id in train_dataset:
    xs.append(x)
    ids.append(_id)
x_lens = [len(x) for x in xs]

segment_cnts = [Counter(), Counter(), Counter()]
for x in xs:
    segments = np.array_split(x, n_components) 
    for i, c in enumerate(segments):
        segment_cnts[i] += Counter(c.tolist())

print('train data vocab size = ', len(vocab))
emissionprobs = []
for segment_cnt in segment_cnts:
    emissionprob = [1] * len(cnt)
    for k, v in segment_cnt.items():
        emissionprob[k] = v
    emissionprobs.append(emissionprob)

emissionprobs = np.array(emissionprobs)
emissionprobs = emissionprobs / np.sum(emissionprobs, axis=1, keepdims=True)
print("entropy of emissionprobs")
print(entropy(emissionprobs))
#emissionprob = np.vstack([emissionprob]*n_components)

xs = np.concatenate(xs).reshape(-1, 1).astype(int)

# Build an HMM instance and set parameters
model = hmm.MultinomialHMM(n_components=n_components, init_params="",
                           n_iter=n_iter, verbose=True)

model.startprob_ = startprob
model.transmat_ = transmat
model.emissionprob_ = emissionprobs

model._do_mstep = _do_mstep.__get__(model, _do_mstep)

print("training ...")
model = model.fit(xs, x_lens)

state_info = [str(i) for i in range(n_components)]
emission_entropy = entropy(model.emissionprob_)
plot_bar(state_info, emission_entropy, f"./images/entropy/{len(state_info)}.eps")


for s in range(max_num_splits):

    # state-splitting
    n_components += 1
    e = entropy(model.emissionprob_)
    split_idx = np.argmax(e)

    state_info.append(f"{state_info[split_idx]}_s")

    old_model = copy.deepcopy(model)
    startprob, _ = split_state_startprob(model.startprob_, split_idx)
    transmat, transmat_mask = split_state_transmat(model.transmat_, split_idx)
    emissionprob, _ = split_state_emission(model.emissionprob_, split_idx)
    
    print("************ before splitting ************")
    #print("startprob")
    #print(model.startprob_)
    #print("transmat")
    #print(model.transmat_)
    print("entropy")
    print(e)
    print("avg of entropy")
    print(np.mean(e))

    # retrain model again
    model = hmm.MultinomialHMM(n_components=n_components, init_params="",
                               n_iter=n_iter, verbose=True)

    _do_mstep = partial(_do_mstep, t_mask=transmat_mask)
    funcType = types.MethodType
    model._do_mstep = funcType(_do_mstep, model) 

    model.startprob_ = startprob
    model.transmat_ = transmat
    model.emissionprob_ = emissionprob

    #print("************ after splitting ************")
    #print("startprob")
    #print(model.startprob_)
    #print("transmat")
    #print(model.transmat_)
    
    print()
    print(f"training ...")
    print(f"split_idx = {split_idx}; n_components = {n_components}")
    
    model = model.fit(xs, x_lens)

    print("************ after training ************")
    print(f"last logprob = {old_model.monitor_.history[-1]}")
    print(f"now logprob = {model.monitor_.history[-1]}")
    print(f"last ppl = {old_model.monitor_.history[-1]/np.mean(x_lens)}")
    print(f"now ppl = {model.monitor_.history[-1]/np.mean(x_lens)}")
    emission_entropy = entropy(model.emissionprob_)
    plot_bar(state_info, emission_entropy, f"./images/entropy/{len(state_info)}.eps")
    print("entropy")
    print(emission_entropy)
    print("avg of entropy")
    print(np.mean(emission_entropy))

    print()
    print("#"*20, " epoch end ", "#"*20)
    print()

    with open(os.path.join(exp_dir, f"{n_components}.pkl"), "wb") as f:
        model._do_mstep = None
        pickle.dump(model, f)
