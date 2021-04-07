import numpy as np
from collections import Counter, OrderedDict, defaultdict
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

    a = set(s.name for s in states)
    b = set(s.name for s in model.states)
    assert a == b, f'a = {a} \n b = {b}'

    return model


def init_model(xs, num_init_states, count, init_threshold):

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

    # Define the distribution for insertion states
    for k, v in sorted(count.items(), key=lambda x: x[1], reverse=True)[:4]:
        count[k] = 1
    for k in [140, 114, 142, 45]:
        count[k] = 1
    global_emission_prob = normalize(count)
    i_d = DiscreteDistribution(global_emission_prob)
    states = []
    insert_states = []
    for k in range(num_init_states):
        s_name = f"S{k+1:02}"
        s = State(DiscreteDistribution(emissionprobs[k]), name=s_name)
        states.append(s)

        i_name = f"I@{s_name}"
        # tie distribution
        s = State(i_d, name=i_name)
        insert_states.append(s)

    model = HiddenMarkovModel(name='Conversation HMM')

    state2child = {}
    for i in range(num_init_states-1):
        state2child[states[i].name] = states[i+1].name
    state2child[states[-1].name] = model.end.name

    edges = []
    # Create transitions from match states
    edges.append((model.start, states[0], 1.0))

    self_loop_prob = 0.75
    insert_prob = 0.05
    for i in range(num_init_states):

        # self loop
        edges.append((states[i], states[i], self_loop_prob))
        edges.append((insert_states[i], insert_states[i], self_loop_prob))
        edges.append((states[i], insert_states[i], insert_prob))
        edges.append((insert_states[i], states[i], insert_prob))

        num_neighbors = min(2, num_init_states-i)
        # +1 is for the end state
        neighbor_prob = (1-self_loop_prob-insert_prob) / (num_neighbors+1)
        for j in range(i+1, i+num_neighbors):
            edges.append((states[i], states[j], neighbor_prob))
            edges.append((insert_states[i], states[j], neighbor_prob))

        # each state can go to the end state
        edges.append((states[i], model.end, neighbor_prob))
        edges.append((insert_states[i], model.end, neighbor_prob))

    states.extend([model.start, model.end])
    states.extend(insert_states)
    model = create_model(states, edges, model)

    return model, state2child


def get_topk_outgoing(named_edges, end_state, topk):
    i2j = defaultdict(list)
    new_named_edges = []
    for i, j, prob in named_edges:
        if j == end_state:
            new_named_edges.append((i, j, prob))
        else:
            i2j[i].append((j, prob))

    for i, nodes in i2j.items():
        for j, prob in sorted(nodes, key=lambda x: x[1], reverse=True)[:topk]:
            new_named_edges.append((i, j, prob))
    return new_named_edges
    

def temporal_split(model_json, split_state, state2child, num_temporal_split, topk_outgoing=0):
    named_edges, insert_named_edges = get_named_edges(model_json)
    state2emissionprob, insert_state2emissionprob, _ = get_states(model_json)

    new = f'T{num_temporal_split+1:02} <- {split_state}'
    insert_new = f'I@{new}'
    child = state2child[split_state]
    state2child[new] = child

    model = HiddenMarkovModel(name='Conversation HMM')

    # n2s: name2state
    global_emission_prob = {int(k): v for k, v in insert_state2emissionprob['I@S01'].items()}
    i_d = DiscreteDistribution(global_emission_prob)
    n2s = {}
    for name, ep in state2emissionprob.items(): 
        ep = {int(k): v for k, v in ep.items()}
        n2s[name] = State(DiscreteDistribution(ep), name=name)
        n2s[f'I@{name}'] = State(i_d, name=f'I@{name}')
        if name == split_state:
            n2s[new] = State(DiscreteDistribution(ep), name=new)
            n2s[insert_new] = State(i_d, name=insert_new)
    n2s[model.start.name] = model.start
    n2s[model.end.name] = model.end

    # temporal split
    new_named_edges = []
    for i, j, prob in named_edges:
        if i == split_state:
            if j == child:
                # temporal split
                new_named_edges.append((n2s[i], n2s[j], prob/2))
                new_named_edges.append((n2s[i], n2s[new], prob/2))
                new_named_edges.append((n2s[new], n2s[j], prob))
            elif j == f'I@{i}':
                new_named_edges.append((n2s[new], n2s[insert_new], prob))
                new_named_edges.append((n2s[i], n2s[j], prob))
            else:
                # j might equal to i so already include self loop
                new_named_edges.append((n2s[i], n2s[j], prob))
                # self loop
                if i == j:
                    new_named_edges.append((n2s[new], n2s[new], prob))
                else:
                    new_named_edges.append((n2s[new], n2s[j], prob))
        else:
            new_named_edges.append((n2s[i], n2s[j], prob))


    for i, j, prob in insert_named_edges:
        normal_state_name = i.split('@')[1]
        if normal_state_name == split_state:
            if i == j:
                new_named_edges.append((n2s[insert_new], n2s[insert_new], prob))
            elif j == normal_state_name:
                new_named_edges.append((n2s[insert_new], n2s[new], prob))
            else:
                new_named_edges.append((n2s[insert_new], n2s[j], prob))
        new_named_edges.append((n2s[i], n2s[j], prob))

    if topk_outgoing:
        new_named_edges = get_topk_outgoing(new_named_edges, model.end, topk_outgoing)

    model = create_model(list(n2s.values()), new_named_edges, model)

    return model, state2child


def vertical_split(model_json, split_state, state2child, num_vertical_split, topk_outgoing):
    named_edges, insert_named_edges = get_named_edges(model_json)
    state2emissionprob, insert_state2emissionprob, _ = get_states(model_json)

    new = f'V{num_vertical_split+1:02} <- {split_state}'
    insert_new = f'I@{new}'
    child = state2child[split_state]
    state2child[new] = child

    model = HiddenMarkovModel(name='Conversation HMM')

    # n2s: name2state
    global_emission_prob = {int(k): v for k, v in insert_state2emissionprob['I@S01'].items()}
    i_d = DiscreteDistribution(global_emission_prob)
    n2s = {}
    for name, ep in state2emissionprob.items(): 
        ep = {int(k): v for k, v in ep.items()}
        n2s[name] = State(DiscreteDistribution(ep), name=name)
        n2s[f'I@{name}'] = State(i_d, name=f'I@{name}')
        if name == split_state:
            max_prob_cluster = max(ep.items(), key=lambda x: x[1])[0]
            ep[max_prob_cluster] = 0
            n2s[new] = State(DiscreteDistribution(ep), name=new)
            n2s[insert_new] = State(i_d, name=insert_new)
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
            elif j == f'I@{i}':
                new_named_edges.append((n2s[new], n2s[insert_new], prob))
                new_named_edges.append((n2s[i], n2s[j], prob))
            else:
                # j might equal to i so already include self loop
                new_named_edges.append((n2s[i], n2s[j], prob))
                # self loop
                if i == j:
                    new_named_edges.append((n2s[new], n2s[new], prob))
                else:
                    new_named_edges.append((n2s[new], n2s[j], prob))
                #new_named_edges.append((n2s[i], n2s[j], prob))
                #new_named_edges.append((n2s[new], n2s[j], prob))
        elif j == split_state:
            new_named_edges.append((n2s[i], n2s[j], prob/2))
            new_named_edges.append((n2s[i], n2s[new], prob/2))
        else:
            new_named_edges.append((n2s[i], n2s[j], prob))

    for i, j, prob in insert_named_edges:
        normal_state_name = i.split('@')[1]
        if normal_state_name == split_state:
            # self loop
            if i == j:
                new_named_edges.append((n2s[insert_new], n2s[insert_new], prob))
            elif j == normal_state_name:
                new_named_edges.append((n2s[insert_new], n2s[new], prob))
            else:
                new_named_edges.append((n2s[insert_new], n2s[j], prob))
        new_named_edges.append((n2s[i], n2s[j], prob))

    if topk_outgoing:
        new_named_edges = get_topk_outgoing(new_named_edges, model.end, topk_outgoing)

    model = create_model(list(n2s.values()), new_named_edges, model)

    return model, state2child


def plot_shmm(model, state2child, image_path):
    df = pd.read_csv('./raw_data/150/medoid_centers.csv')
    df['cluster'] = df['cluster'].astype(str)
    cluster2utt = dict(zip(df.cluster, df.phrase))

    emission_topk = 5
    model_json = json.loads(model.to_json())

    state2emissionprob, insert_state2emissionprob, dummy_states = get_states(model_json)

    g = Digraph('G', engine="dot")

    c0 = Digraph('match')
    c0.body.append('style=filled')
    c0.body.append('color=white')
    c0.attr('node', shape='box')
    c0.node_attr.update(color='#D0F0FB', style='filled',fontsize="14")
    #c0.edge_attr.update(color='#076789', fontsize="12")
    c0.edge_attr.update(color='white')

    c1 = Digraph('insert')
    c1.body.append('style=filled')
    c1.body.append('color=white')
    c1.attr('node', shape='box')
    c1.node_attr.update(color='#AFF3BA', style='filled',fontsize="14")
    #c1.edge_attr.update(color='#FFE200', fontsize="12")
    c1.edge_attr.update(color='white')

    """
        edges
    """
    idx2names = {i: s['name'] for i, s in enumerate(model_json['states'])}
    i2js = defaultdict(list)
    for e in model_json['edges']:
        i = idx2names[e[0]]
        j = idx2names[e[1]]
        prob = e[2]
        i2js[i].append((j, prob))

    """
        emission
    """
    state2topk_clusters = OrderedDict()
    sn2ep = list(state2emissionprob.items()) + list(insert_state2emissionprob.items())
    for name, ep in sn2ep:
        if name[0] == 'I':
            _emission_topk = 20
        else:
            _emission_topk = emission_topk
        string = name + "\n"
        topk = sorted(ep.items(), key=lambda x: x[1], reverse=True)[:_emission_topk]
        for cluster, prob in topk:
            string += f'cluster = {cluster}; prob = {prob:.4f}\l'
            string += "\l".join(textwrap.wrap(f"{cluster2utt[cluster]}", width=40))
            string += "\l\l"
        state2topk_clusters[name] = string


    """
        start adn end
    """
    for s in dummy_states:
        c0.node(s)

    for j, _ in i2js[model.start.name]:
        c0.edge(model.start.name, j)

    """
        normal states
    """
    for name in state2emissionprob.keys():
        topk_clusters = state2topk_clusters[name]
        c0.node(name, topk_clusters)
        # for better visualization, we tie child with parent by a white edge
        c0.edge(name, state2child[name])


    """
        insert states
    """
    for name in insert_state2emissionprob.keys():
        topk_clusters = state2topk_clusters[name]
        c1.node(name, topk_clusters)

        # for better visualization, we tie child with parent by a white edge
        normal_state_name = name.split('@')[1]
        child_name = state2child[normal_state_name]
        if child_name != model.end.name:
            child_insert_state_name = f'I@{child_name}'
            c1.edge(name, child_insert_state_name)
        
    #the graph is basicaly split in 4 clusters
    g.subgraph(c0)
    g.subgraph(c1)

    # sort edges by their prob
    for i, js in i2js.items():
        i2js[i] = sorted(js, key=lambda x: x[1], reverse=True)

    for i in list(state2emissionprob.keys()) + list(insert_state2emissionprob.keys()) + [model.start.name]:
        red_top1 = False
        if i[0] == 'I':
            color = '#388B46'
        else:
            color = '#005FFE'
        js = i2js[i]
        for k, (j, prob) in enumerate(js):
            if j == model.end.name:
                g.edge(i, j, label=f'{prob:.3f}', len='3.00', style='dashed', color=color)
            elif i != j and not red_top1 and i[0] != 'I':
                g.edge(i, j, label=f'{prob:.3f}', len='3.00', color='#F0644D')
                red_top1 = True
            else:
                g.edge(i, j, label=f'{prob:.3f}', len='3.00', color=color)


    g.edge_attr.update(arrowsize='1')
    g.body.extend(['rankdir=LR', 'size="159,100"'])
    g.render(image_path, format='pdf')
