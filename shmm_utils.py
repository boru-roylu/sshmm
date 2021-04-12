import numpy as np
from collections import Counter, OrderedDict
from graphviz import Digraph
from pomegranate import *
import textwrap

from utils import get_states, get_named_edges 

import pandas as pd

def normalize(dic):
    _sum = sum(dic.values())
    return {k: v/_sum for k, v in dic.items()}


def create_model(states, edges, model):
    model.add_states(states)
    for e in edges:
        model.add_transition(*e)
    # Call bake to finalize the structure of the model.
    model.bake()

    return model


def init_model(xs, num_init_states, count, init_threshold):
    # Define the distribution for insertion states
    global_emission_probs = normalize(count)

    # Define the distribution for match states
    segment_ratios = [0.05]
    segment_ratios += [0.90/(num_init_states-2)]*(num_init_states-2)
    segment_ratios += [0.05]
    segment_ratios = np.array(segment_ratios)
    segment_cnts = [Counter() for _ in range(num_init_states)]
    for x in xs:
        if len(x) < init_threshold:
            continue
        _segment_ratios = np.ceil(segment_ratios * len(x)).astype(int)
        _split_points = np.cumsum(_segment_ratios)
        if _split_points[-1] >= len(x):
            _split_points = _split_points[:-1]
        segments = np.split(x, _split_points)
        for i, c in enumerate(segments):
            segment_cnts[i] += Counter(c.tolist())

    emissionprobs = []
    for segment_cnt in segment_cnts:
        eprobs = {k: 0 for k in count.keys()}
        eprobs.update(segment_cnt)
        eprobs = normalize(eprobs)
        emissionprobs.append(eprobs)


    states = []
    for k in range(num_init_states):
        name = f"S{k+1:02}"
        s = State(DiscreteDistribution(emissionprobs[k]), name=name)
        states.append(s)

    model = HiddenMarkovModel(name="Conversation HMM")

    state2child = {}
    for i in range(num_init_states-1):
        state2child[states[i].name] = states[i+1].name
    state2child[states[-1].name] = model.end

    edges = []
    # Create transitions from match states
    edges.append((model.start, states[0], 1.0))

    self_loop_prob = 0.8
    for i in range(num_init_states):

        # self loop
        edges.append((states[i], states[i], self_loop_prob))

        num_neighbors = min(2, num_init_states-i)
        # +1 is for the end state
        neighbor_prob = (1-self_loop_prob) / (num_neighbors+1)
        for j in range(i+1, i+num_neighbors):
            edges.append((states[i], states[j], neighbor_prob))

        # each state can go to the end state
        edges.append((states[i], model.end, neighbor_prob))

    model = create_model(states, edges, model)

    return model, state2child


def temperal_split(model_json, split_state, state2child, num_temperal_split):
    named_edges = get_named_edges(model_json)
    state2emissionprob, _ = get_states(model_json)

    new = f"T{num_temperal_split+1:02} <- {split_state}"
    child = state2child[split_state]
    state2child[new] = child

    model = HiddenMarkovModel(name="Conversation HMM")

    # n2s: name2state
    n2s = {}
    for name, ep in state2emissionprob.items(): 
        ep = {int(k): v for k, v in ep.items()}
        n2s[name] = State(DiscreteDistribution(ep), name=name)
        if name == split_state:
            n2s[new] = State(DiscreteDistribution(ep), name=new)
    n2s[model.start.name] = model.start
    n2s[model.end.name] = model.end

    # temperal split
    new_named_edges = []
    for i, j, prob in named_edges:
        if i == split_state:
            if j == child:
                new_named_edges.append((n2s[i], n2s[j], prob/2))
                new_named_edges.append((n2s[i], n2s[new], prob/2))
                new_named_edges.append((n2s[new], n2s[j], prob))
            else:
                new_named_edges.append((n2s[i], n2s[j], prob))
                if i == j:
                    new_named_edges.append((n2s[new], n2s[new], prob))
                else:
                    new_named_edges.append((n2s[new], n2s[j], prob))
        else:
            new_named_edges.append((n2s[i], n2s[j], prob))

    model = create_model(list(n2s.values()), new_named_edges, model)

    return model, state2child


def vertical_split(model_json, split_state, state2child, num_vertical_split, count):
    global_emission_probs = normalize(count)

    named_edges = get_named_edges(model_json)
    state2emissionprob, _ = get_states(model_json)

    new = f"V{num_vertical_split+1:02} <- {split_state}"
    child = state2child[split_state]
    state2child[new] = child

    model = HiddenMarkovModel(name="Conversation HMM")

    # n2s: name2state
    n2s = {}
    for name, ep in state2emissionprob.items(): 
        ep = {int(k): v for k, v in ep.items()}
        n2s[name] = State(DiscreteDistribution(ep), name=name)
        if name == split_state:
            max_prob_cluster = max(ep.items(), key=lambda x: x[1])[0]
            ep[max_prob_cluster] = 0
            #n2s[new] = State(DiscreteDistribution(ep), name=new)
            n2s[new] = State(DiscreteDistribution(global_emission_probs), name=new)
    n2s[model.start.name] = model.start
    n2s[model.end.name] = model.end

    # vertical split
    new_named_edges = []
    for i, j, prob in named_edges:
        if i == split_state:
            # self loop
            if j == split_state:
                new_named_edges.append((n2s[i], n2s[i], prob))
                new_named_edges.append((n2s[new], n2s[new], prob))
            else:
                new_named_edges.append((n2s[i], n2s[j], prob))
                new_named_edges.append((n2s[new], n2s[j], prob))

        elif j == split_state:
            new_named_edges.append((n2s[i], n2s[j], prob/2))
            new_named_edges.append((n2s[i], n2s[new], prob/2))
        else:
            new_named_edges.append((n2s[i], n2s[j], prob))

    model = create_model(list(n2s.values()), new_named_edges, model)

    return model, state2child


def plot_shmm(model, image_path):
    df = pd.read_csv('./raw_data/150/medoid_centers.csv')
    df['cluster'] = df['cluster'].astype(str)
    cluster2utt = dict(zip(df.cluster, df.phrase))

    emission_topk = 5
    model_json = json.loads(model.to_json())

    state2emissionprob, dummy_states = get_states(model_json)

    g = Digraph('G', engine="dot")

    # match emission
    c0 = Digraph('cluster_0')
    c0.body.append('style=filled')
    c0.body.append('color=white')
    c0.attr('node', shape='box')
    c0.node_attr.update(color='#D0F0FB', style='filled',fontsize="14")
    c0.edge_attr.update(color='#076789', fontsize="12")

    state2topk_clusters = OrderedDict()
    for name, ep in state2emissionprob.items():
        string = name + "\n"
        topk = sorted(ep.items(), key=lambda x: x[1], reverse=True)[:emission_topk]
        for cluster, prob in topk:
            string += f'cluster = {cluster}; prob = {prob:.4f}\l'
            string += "\l".join(textwrap.wrap(f"{cluster2utt[cluster]}", width=40))
            string += "\l\l"
        state2topk_clusters[name] = string


    for s in dummy_states:
        c0.node(s)
    for name, topk_clusters in state2topk_clusters.items():
        c0.node(name, topk_clusters)

    # edge
    idx2names = {i: s['name'] for i, s in enumerate(model_json['states'])}
    edges = []
    for e in model_json['edges']:
        i_state = idx2names[e[0]]
        j_state = idx2names[e[1]]
        prob = e[2]
        c0.edge(i_state, j_state, label=f"{prob:.3f}", len='3.00')

    #the graph is basicaly split in 4 clusters
    g.subgraph(c0)

    g.edge_attr.update(arrowsize='1')
    g.body.extend(['rankdir=LR', 'size="160,100"'])
    g.render(image_path, format='pdf')
