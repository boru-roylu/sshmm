import numpy as np
from hmmlearn.utils import normalize
from hmmlearn import hmm

def _do_mstep(self, stats, s_mask=None, t_mask=None, e_mask=None):
    if 's' in self.params:
        startprob_ = np.maximum(self.startprob_prior - 1 + stats['start'], 0)
        self.startprob_ = np.where(self.startprob_ == 0, 0, startprob_)
        normalize(self.startprob_)

    if 't' in self.params:
        transmat_ = np.maximum(self.transmat_prior - 1 + stats['trans'], 0)
        transmat_ = np.where(t_mask == 1, transmat_, self.transmat_)
        self.transmat_ = np.where(self.transmat_ == 0, 0, transmat_)
        normalize(self.transmat_, axis=1)

    if 'e' in self.params:
        self.emissionprob_ = (
            stats['obs'] / stats['obs'].sum(axis=1, keepdims=True))


def split_state_startprob(old, i):
    new = []
    for j, p in enumerate(old):
        if i == j:
            p /= 2
        new.append(p)
    new.append(new[i])
    new = np.array(new)
    return new, None


def split_state_transmat(old, i):
    new = []
    mask = []
    for j, p in enumerate(old):
        p = p.tolist()
        if i != len(old) and i == j:
            if i+1 != len(old):
                p[i+1] /= 2
                p.append(p[i+1])
            else:
                p[i] /= 2
                p.append(p[i])
            m = [1]*len(p)
        else:
            p.append(0)
            m = [0] * (len(p)-1) + [1]
        mask.append(m)
        new.append(p)

    if i == len(old):
        new[i][j+1] = new[i][j] / 2
        new[i][j] /= 2

    new.append(new[i])
    mask.append(mask[i])
    new = np.array(new)
    mask = np.array(mask)

    new[-1][-1] += new[-1][i]
    new[-1][i] = 0

    return new, mask


def split_state_emission(old, i):
    b = ((old[i-1] + old[i]) / 2).tolist()
    new = old.tolist()
    new.append(b)
    new = np.array(new)
    return new, None


def entropy(prob):
    log_prob = np.log(np.clip(prob, 1e-12, 1))
    e = -np.sum(prob * log_prob, axis=1)
    return e
