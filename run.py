"""
Sampling from HMM
-----------------

This script shows how to sample points from a Hidden Markov Model (HMM):
we use a 4-state model with specified mean and covariance.

The plot show the sequence of observations generated with the transitions
between them. We can see that, as specified by our transition matrix,
there are no transition between component 1 and 3.
"""

import numpy as np
from sklearn.utils import check_random_state

from hmmlearn import hmm
from hmmlearn.utils import normalize

from data import get_datasets
from sshmm import _do_mstep, split_state_startprob, split_state_transmat, split_state_emission

train_dataset, dev_dataset, vocab = get_datasets("./data/agent")

print('vocab size = ', len(vocab))

random_state=42
n_components = 4
init_first_state_prob = 0.99
n_features = len(vocab)

##############################################################
# Prepare parameters for a 4-components HMM
# Initial population probability
startprob = np.zeros(n_components)
startprob[0] = init_first_state_prob
startprob[1:] = (1-init_first_state_prob) / (n_components-1)

# The transition matrix, note that there are no transitions possible
# between component 1 and 3
transmat = np.array([[0.7, 0.2, 0.0, 0.1],
                     [0.3, 0.5, 0.2, 0.0],
                     [0.0, 0.3, 0.5, 0.2],
                     [0.2, 0.0, 0.2, 0.6]])


rs = check_random_state(random_state)
emissionprob = rs.rand(n_components, n_features)
normalize(emissionprob, axis=1)

xs = list(iter(train_dataset))
x_lens = [len(x) for x in xs]

xs = np.concatenate(xs).reshape(-1, 1)

# Build an HMM instance and set parameters
model = hmm.MultinomialHMM(n_components=n_components,
                           init_params="",
                           n_iter=1, verbose=True)

model.startprob_ = startprob
model.transmat_ = transmat
model.emissionprob_ = emissionprob

model._do_mstep = _do_mstep.__get__(model, _do_mstep)

print("training ...")
model = model.fit(xs, x_lens)

print(model.emissionprob_.shape)
print(model.emissionprob_.shape)
print(model.emissionprob_.shape)
exit()
split_idx = 0
n_components += 1

startprob = split_state_startprob(model.startprob_, split_idx)
transmat = split_state_transmat(model.transmat_, split_idx)
emissionprob = split_state_emission(model.emissionprob_, split_idx)


model = hmm.MultinomialHMM(n_components=n_components,
                           init_params="",
                           n_iter=1, verbose=True)
model._do_mstep = _do_mstep.__get__(model, _do_mstep)

model.startprob_ = startprob
model.transmat_ = transmat
model.emissionprob_ = emissionprob

print("training ...")
model.fit(xs, x_lens)

# Generate samples
#X, Z = model.sample(500)
