"""
Sampling from HMM
-----------------

This script shows how to sample points from a Hidden Markov Model (HMM):
we use a 4-state model with specified mean and covariance.

The plot show the sequence of observations generated with the transitions
between them. We can see that, as specified by our transition matrix,
there are no transition between component 1 and 3.
"""

import copy
import pickle
import types
import numpy as np
from sklearn.utils import check_random_state
from functools import partial

from hmmlearn import hmm
from hmmlearn.utils import normalize

from data import get_datasets
from sshmm import _do_mstep, split_state_startprob, split_state_transmat, split_state_emission, entropy

topk_cluster = 30
train_dataset, dev_dataset, vocab = get_datasets("./data/agent", topk_cluster)

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
                     [0.0, 0.0, 1.0],])


#rs = check_random_state(random_state)
#emissionprob = rs.rand(n_components, n_features)
#normalize(emissionprob, axis=1)

xs = list(iter(train_dataset))
x_lens = [len(x) for x in xs]

vocab2freq = {}
for x in xs:
    for xx in x:
        vocab2freq[xx] = vocab2freq.get(xx, 0) + 1

print('train data vocab size = ', len(vocab2freq))
emissionprob = [0] * len(vocab2freq)
for k, v in sorted(vocab2freq.items(), key=lambda x: x[0]):
    emissionprob[k] = v
emissionprob = np.array(emissionprob) / sum(emissionprob)
emissionprob = np.vstack([emissionprob]*n_components)

xs = np.concatenate(xs).reshape(-1, 1).astype(int)

# Build an HMM instance and set parameters
model = hmm.MultinomialHMM(n_components=n_components, init_params="",
                           n_iter=n_iter, verbose=True)

model.startprob_ = startprob
model.transmat_ = transmat
model.emissionprob_ = emissionprob

model._do_mstep = _do_mstep.__get__(model, _do_mstep)

print("training ...")
model = model.fit(xs, x_lens)


for s in range(max_num_splits):

    # state-splitting
    n_components += 1
    e = entropy(model.emissionprob_)
    split_idx = np.argmax(e)

    old_model = copy.deepcopy(model)
    startprob, _ = split_state_startprob(model.startprob_, split_idx)
    transmat, transmat_mask = split_state_transmat(model.transmat_, split_idx)
    emissionprob, _ = split_state_emission(model.emissionprob_, split_idx)
    
    print("************ before splitting startprob ************")
    print(model.startprob_)
    print("************ before splitting transmat ************")
    print(model.transmat_)

    # retrain model again
    model = hmm.MultinomialHMM(n_components=n_components, init_params="",
                               n_iter=n_iter, verbose=True)

    _do_mstep = partial(_do_mstep, t_mask=transmat_mask)
    funcType = types.MethodType
    model._do_mstep = funcType(_do_mstep, model) 

    model.startprob_ = startprob
    model.transmat_ = transmat
    model.emissionprob_ = emissionprob

    print("************ after splitting startprob ************")
    print(model.startprob_)
    print("************ after splitting transmat ************")
    print(model.transmat_)
    
    print(f"training ...")
    print(f"split_idx = {split_idx}; n_components = {n_components}")
    
    model = model.fit(xs, x_lens)

    print(f"last logprob = {old_model.monitor_.history[-1]}")
    print(f"now logprob = {model.monitor_.history[-1]}")
    print(f"last ppl = {old_model.monitor_.history[-1]/np.mean(x_lens)}")
    print(f"now ppl = {model.monitor_.history[-1]/np.mean(x_lens)}")

    print()
    print("#"*20, " epoch end ", "#"*20)
    print()

    with open(f"./models/{n_components}.pkl", "wb") as f:
        model._do_mstep = None
        pickle.dump(model, f)
