import os
import json
import copy
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from graphviz import Digraph
import math
import pdb
import textwrap

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
        #model._do_mstep = None
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

def graph_topo(transmat, emissionprob, state_info, vocab, top_e):
    dot = Digraph(comment='State Topo')
    dot.attr(rankdir='LR')
    dot.attr('node', shape='circle')

    # TODO ad hoc
    ordered_idxs = get_ordered_idxs(state_info, 3)

    for i, idx in enumerate(ordered_idxs):
        ep = emissionprob[idx]
        top_probs_idx = np.argsort(ep, )[-top_e:][::-1]
        top_probs_clusters = [vocab[t] for t in top_probs_idx]
        rep_utts = get_cluster_representative(top_probs_clusters)
        children_list = [str(s) for s in trace(state_info, idx)][1:]
        to_display = f"state: {idx} childen: {','.join(children_list)} \n"
        #parent = str(state_info[idx])
        #to_display = f"state: {idx} ch: {parent}\n"
        for j, p in enumerate(top_probs_idx):
            utt = rep_utts[j].split()
            utt_split = np.array_split(utt, math.ceil(len(utt) / 10))
            utt = "\l".join([' '.join(s) for s in list(utt_split)])
            to_display += f"cluster {vocab[p]} - " + "{:.5f}".format(ep[p]) + f":\l{utt}\l\n"
        dot.node(f"{i:02}",
                 f"{to_display}",
                 color="purple", fillcolor='#E6E6FA', style='filled', shape='box')

    for i, idx in enumerate(ordered_idxs):
        for jdx in range(transmat.shape[0]):
            j = ordered_idxs.index(jdx)
            prob = transmat[idx, jdx]
            if prob > 0:
                dot.edge(f"{i:02}",
                         f"{j:02}",
                         label = f"{prob:.3f}",
                         color='purple', penwidth="1.0")
    dot.render('/g/ssli/data/tmcalls/sshmm/transmat.gv', format='pdf', view=False)




    
    #for t in range(transmat.shape[1]-1, -1, -1):
    #    for r in range(transmat.shape[0]):
    #        if transmat[r, t] != 0.0:
    #            ep = emissionprob[t]
    #            top_probs_idx = np.argsort(ep, )[-top_e:][::-1]
    #            top_probs_clusters = [vocab[idx] for idx in top_probs_idx]
    #            rep_utts = get_cluster_representative(top_probs_clusters)
    #            to_display = ""
    #            for i, p in enumerate(top_probs_idx):
    #                utt = rep_utts[i].split()
    #                utt_split = np.array_split(utt, math.ceil(len(utt) / 10))
    #                utt = "\l".join([' '.join(s) for s in list(utt_split)])
    #                to_display += f"cluster {vocab[p]} - " + "{:.5f}".format(ep[p]) + f":\l{utt}\l\n"
    #            dot.node(f"{t}",
    #                     f"{to_display}",
    #                    color="purple", fillcolor='#E6E6FA', style='filled', shape='box')
    #            dot.node(f"{r}",
    #                    color="purple", fillcolor='#E6E6FA', style='filled', shape='box')
    #            dot.edge(f"{r}",
    #                     f"{t}",
    #                     label = "{:.3f}".format(transmat[r,t]),
    #                     color='purple', penwidth="1.0")

    #dot.render('./sshmm', format='pdf', view=False)

def get_cluster_representative(cluster_ids):
    df = pd.read_csv('/g/ssli/data/tmcalls/clustering_data/clustering/medoid_centers.csv')
    df = df[df.cluster.isin(cluster_ids)].reindex(cluster_ids)
    rep_utts = df.phrase.tolist()
    return rep_utts


def plotHMM(model):
    """ Plot a profile-HMM based on its structure using graphviz.
    There's not guarantee that the ouput would be visualy informative for
    HMMs with many states.
    Args:
        edges (dict): edge weights between states
        match_ids (list): list of ids that would correspond match states
        delete_ids (list): list of ids that would correspond delete states
        insert_ids (list): list of ids that would correspond insert states
        match_emissionprobs (dict): dictionary of the form {state_id:{cluster:float,clusterB: float}}
    """

    df = pd.read_csv('/g/ssli/data/tmcalls/clustering_data/clustering/medoid_centers.csv')
    df['cluster'] = df['cluster'].astype(str)
    cluster2utt = dict(zip(df.cluster, df.phrase))

    match_topk = 5
    insert_topk = 5

    dic = json.loads(model.to_json())
    match_ids, delete_ids, insert_ids = [], [], []
    match_emissionprobs = {}
    insert_emissionprobs = {}
    for s in dic['states']:
        n = s['name']
        if 'D' in n:
            delete_ids.append(n)
        elif 'I' in n:
            insert_ids.append(n)
            insert_emissionprobs[n] = s['distribution']['parameters'][0]
        else:
            match_ids.append(n)
            if 'start' in n or 'end' in n:
                continue
            match_emissionprobs[n] = s['distribution']['parameters'][0]

    idx2names = {i: s['name'] for i, s in enumerate(dic['states'])}
    edges = []
    for e in dic['edges']:
        edges.append((idx2names[e[0]], idx2names[e[1]], e[2]))

    g = Digraph('G', filename='cluster.gv', format="pdf", engine="dot")

    c0 = Digraph('cluster_0')
    c0.body.append('style=filled')
    c0.body.append('color=white')
    c0.attr('node', shape='box')
    c0.node_attr.update(color='orange', style='filled')

    match_ids_without_start_end = match_ids[:]
    index = match_ids.index("Conversation HMM-end")
    match_ids_without_start_end.pop(index)
    index = match_ids.index("Conversation HMM-start")
    match_ids_without_start_end.pop(index)

    for match_id in match_ids:
        c0.node(match_id)

    c1 = Digraph('cluster_1')
    c1.body.append('style=filled')
    c1.body.append('color=white')
    c1.attr('node', shape='doubleoctagon')
    c1.node_attr.update(color="orange", penwidth="1")
    for insert_id in insert_ids:
        c1.node(insert_id)
    c1.edge_attr.update(color='white')
    for i in range(len(insert_ids)-1):
        c1.edge(insert_ids[i], insert_ids[i+1])

    c2 = Digraph('cluster_2')
    c2.body.append('style=filled')
    c2.body.append('color=white')
    c2.attr('node', shape='circle')
    c2.node_attr.update(color="orange", penwidth="2")
    for delete_id in delete_ids:
        c2.node(delete_id)

    # match emission
    c3 = Digraph('cluster_3')
    c3.body.append('style=filled')
    c3.body.append('color=white')
    c3.attr('node', shape='box')
    c3.node_attr.update(color='white', style='filled',fontsize="14")

    mids = [""]
    for m in match_ids_without_start_end:
        s = [f"{m}"]
        for cluster, prob in sorted(match_emissionprobs[m].items(), key=lambda x: x[1], reverse=True)[:match_topk]:
            s.append(f'cluster = {cluster}; prob = {prob:.4f}')
            s.append("\l".join(textwrap.wrap(f"{cluster2utt[cluster]}", width=40)))
        mids.append("\l".join(s))

    c3.edge_attr.update(color='white')
    for mid in mids:
        c3.node(mid)
    for i in range(len(mids)-1):
        c3.edge(mids[i], mids[i+1])

    # insert emission
    c4 = Digraph('cluster_4')
    c4.body.append('style=filled')
    c4.body.append('color=white')
    c4.attr('node', shape='box')
    c4.node_attr.update(color='white', style='filled',fontsize="14")

    iids = [""]
    for i in insert_ids:
        s = [f"{i}"]
        for cluster, prob in sorted(insert_emissionprobs[i].items(), key=lambda x: x[1], reverse=True)[:insert_topk]:
            s.append(f'cluster = {cluster}; prob = {prob:.4f}')
            s.append("\l".join(textwrap.wrap(f"{cluster2utt[cluster]}", width=40)))
        iids.append("\l".join(s))

    c4.edge_attr.update(color='white')
    for iid in iids:
        c3.node(iid)
    for i in range(len(iids)-1):
        c3.edge(iids[i], iids[i+1])


    #the graph is basicaly split in 4 clusters
    g.subgraph(c4)
    g.subgraph(c1)
    g.subgraph(c2)
    g.subgraph(c3)
    g.subgraph(c0)

    #add edges
    for h in edges:
        g.edge(h[0], h[1], label=f"{h[2]:.3f}", len='3.00')

    #g.node('Global Sequence Aligner-start', shape='box')
    #g.node('Global Sequence Aligner-end', shape='box')
    g.edge_attr.update(arrowsize='0.5')
    g.body.extend(['rankdir=LR', 'size="160,100"'])
    g.render('123', format='pdf')



if __name__ == "__main__":
    #t = np.arange(16).reshape(4, 4)
    #tt = get_ordered_transmat(t, {0: [], 1: [3], 2: []})
    a = np.random.rand(16).reshape(4, 4)
    print(a)
    r = topk_transmat(a, 3)
    print(r)
