import os
import copy
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from graphviz import Digraph
import pdb

from hmmlearn.utils import normalize


def trace(state_transmat_info, i):
    ret = [i]
    for j in state_transmat_info[i]:
        if j in state_transmat_info:
            ret.extend(trace(state_transmat_info, j))
        else:
            ret.append(j)
    return ret


def get_ordered_idxs(state_transmat_info, num_init_states):
    ordered_idxs = []
    for i in range(num_init_states):
        ordered_idxs.extend(trace(state_transmat_info, i))
    return ordered_idxs


def get_ordered_emission(emissionprob, ordered_idxs):
    emissionprob = copy.deepcopy(emissionprob)
    emissionprob = emissionprob[ordered_idxs]
    return emissionprob


def get_ordered_transmat(transmat, ordered_idxs):
    transmat = copy.deepcopy(transmat)
    transmat = np.transpose(transmat)
    transmat = transmat[ordered_idxs]
    transmat = np.transpose(transmat)
    transmat = transmat[ordered_idxs]
    return transmat


def get_and_plot_ordered_transmat(transmat, ordered_idxs, path, x_labels):
    transmat = copy.deepcopy(transmat)
    ordered_transmat = get_ordered_transmat(transmat, ordered_idxs)
    num_states = len(ordered_transmat)

    df_list = []
    for i in range(num_states):
        for j in range(num_states):
            #df_list.append((x_labels[i], x_labels[j], ordered_transmat[i][j]))
            df_list.append((i, j, ordered_transmat[i][j]))
    df = pd.DataFrame(df_list, columns=["i", "j", "transprob"])

    df = pd.pivot_table(data=df, index='i', values='transprob', columns='j')

    plt.clf()
    sns.heatmap(df, cmap="Reds", annot=True, fmt=".3f", vmin=0, vmax=1)
    plt.savefig(path)
    return ordered_transmat


def plot_bar(x, y, path, hline=None):
    assert len(x) == len(y)
    plt.clf()
    plt.rcParams["axes.labelsize"] = 8
    g = sns.barplot(x="state_idx", y="entropy", data=pd.DataFrame({"state_idx": x, "entropy": y}))
    if hline:
        g.axhline(y=hline, color="red", linestyle='--', label='single_state')
    plt.savefig(path)


def save_model(num_states, model, exp_dir):
    with open(os.path.join(exp_dir, f"{num_states}.pkl"), "wb") as f:
        model._do_mstep = None
        pickle.dump(model, f)


def get_state_labels(state_idx2parent, ordered_idxs):
    ret = []
    for i in ordered_idxs:
        parent = state_idx2parent[i]
        if parent:
            ret.append(f"{i}_{parent}")
        else:
            ret.append(str(i))
    return ret


def topk_transmat(transmat, topk):
    topk_idxs = np.argsort(transmat)[:, -topk:][:, 0]
    for i, t in enumerate(topk_idxs):
        min_value = transmat[i][t]
        m = np.ma.array(transmat[i], mask=transmat[i]<min_value)
        transmat[i] = m.filled(fill_value=0)
        normalize(transmat[i])

    return transmat

def graph_topo(matrix, state_info):
    dot = Digraph(comment='State Topo')
    dot.attr(rankdir='LR')
    dot.attr('node', shape='circle')

    for t in range(matrix.shape[1]-1, -1, -1):
        for r in range(matrix.shape[0]):
            #pdb.set_trace()
            if matrix[r, t] != 0.0:
                prev = matrix[r, t]
                dot.node(f"{t}",
                        color="purple", fillcolor='#E6E6FA', style='filled')
                dot.node(f"{r}",
                        color="purple", fillcolor='#E6E6FA', style='filled')
                dot.edge(f"{r}",
                         f"{t}",
                         label = "{:.3f}".format(matrix[r,t]),
                         color='purple', penwidth="1.0")
                #dot.nod(f"{matrix[

    dot.render('/g/ssli/data/tmcalls/sshmm/transmat.gv', format='pdf', view=False)



if __name__ == "__main__":
    #t = np.arange(16).reshape(4, 4)
    #tt = get_ordered_transmat(t, {0: [], 1: [3], 2: []})

    a = np.random.rand(16).reshape(4, 4)
    print(a)
    r = topk_transmat(a, 3)
    print(r)
