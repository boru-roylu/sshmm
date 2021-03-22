import numpy as np
from hmmlearn.utils import normalize

def _do_mstep(self, stats, s_mask=None, t_mask=None):
    if 's' in self.params:
        startprob_ = np.maximum(self.startprob_prior - 1 + stats['start'],
                                0)
        self.startprob_ = np.where(self.startprob_ == 0, 0, startprob_)
        normalize(self.startprob_)
    if 't' in self.params:
        transmat_ = np.maximum(self.transmat_prior - 1 + stats['trans'], 0)
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
    return np.array(new)


def split_state_transmat(old, i):
    new = []
    print(old)
    for j, p in enumerate(old):
        p = p.tolist()
        if i != len(old) and i == j:
            p[i+1] /= 2
            p.append(p[i+1])
        else:
            p.append(0)
        new.append(p)

    if i == len(old):
        new[i][j+1] = new[i][j] / 2
        new[i][j] /= 2

    new.append(new[i])
    new = np.array(new)
    new[-1][-1] += new[-1][i]
    new[-1][i] = 0
    return np.array(new)


def split_state_emission(old, i):
    b = ((old[i-1] + old[i]) / 2).tolist()
    new = old.tolist()
    new.append(b)
    return np.array(new)
