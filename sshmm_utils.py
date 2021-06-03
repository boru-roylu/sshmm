import os
import json
import yaml
import textwrap
import numpy as np
import pandas as pd

from collections import Counter, OrderedDict, defaultdict
from graphviz import Digraph
from pomegranate import *

from utils import get_states, get_named_edges 

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


class StateSplitingHMM:
    def __init__(self, args, count):
        df = pd.read_csv(args.center_path, sep='|')
        if args.manual_center:
            df.phrase = [f'<{s}> {l}'for s, l in zip(df.stage, df.label)]
        else:
            df.phrase = df['closest_0']#.apply(lambda x: x.replace('numnum', '###'))
        df['cluster'] = df['cluster'].astype(str)
        self.cluster2utt = dict(zip(df.cluster, df.phrase))

        self.args = args
        self.count = count
        self.global_emission_probs = normalize(count)
        self.iteration = 0
        self.num_temperal_split = 0
        self.num_vertical_split = 0

    def save(self, model_dir):
        with open(os.path.join(model_dir, f'sshmm_{self.num_states:03}.json'), 'w') as f:
            json.dump(self.model.to_json(), f)

    @staticmethod
    def load(model_path):
        with open(model_path, 'r') as f:
            model = HiddenMarkovModel.from_yaml(yaml.load(f))
        return model

    @staticmethod
    def fit(model, xs, max_iterations):
        model.fit(
            xs,
            algorithm='baum-welch',
            emission_pseudocount=1,
            stop_threshold=20,
            max_iterations=max_iterations,
            verbose=True,
            n_jobs=os.cpu_count()-2,
        )
        return model

    def fit_split(self, xs, split_state, split_callback):

        model, new_state_name = split_callback(split_state)
        model = StateSplitingHMM.fit(model, xs, self.args.max_iterations)
        logprob = sum(model.log_probability(x) for x in xs)

        return model, logprob, new_state_name


    def init_model(self, xs, num_init_states, init_threshold):
        # Define the distribution for insertion states
        if self.args.insert:
            count = dict(self.count)
            for k in self.args.skip_insert_clusters:
                count[k] = 1
            insert_global_emission_prob = normalize(count)
            i_d = DiscreteDistribution(insert_global_emission_prob)
    
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
            eprobs = {k: 0 for k in self.count.keys()}
            eprobs.update(segment_cnt)
            eprobs = normalize(eprobs)
            emissionprobs.append(eprobs)
    
        states = []
        insert_states = []
        for k in range(num_init_states):
            s_name = f"S{k+1:02}"
            s = State(DiscreteDistribution(emissionprobs[k]), name=s_name)
            states.append(s)

            if self.args.insert:
                i_name = f"I@{s_name}"
                # tie distribution
                s = State(i_d, name=i_name)
                insert_states.append(s)
    
        model = HiddenMarkovModel(name="Conversation HMM")
    
        state2child = {}
        for i in range(num_init_states-1):
            state2child[states[i].name] = states[i+1].name
        state2child[states[-1].name] = model.end.name
    
        edges = []
        # Create transitions from match states
        edges.append((model.start, states[0], 1.0))
    
        self.args.self_loop_prob = 0.8
        for i in range(num_init_states):
            child_name = state2child[states[i].name]
    
            # self loop
            edges.append((states[i], states[i], self.args.self_loop_prob))
            if self.args.insert:
                edges.append((insert_states[i], insert_states[i], self.args.insert_self_loop_epsilon))
    
            num_neighbors = min(2, num_init_states-i)
            # +1 is for the end state
            # +2 is for the end state and insertion state
            neighbor_prob = (1-self.args.self_loop_prob) / {True: num_neighbors+2, False: num_neighbors+1}[self.args.insert]
            if self.args.insert:
                edges.append((states[i], insert_states[i], neighbor_prob))
            for j in range(i+1, i+num_neighbors):
                edges.append((states[i], states[j], neighbor_prob))

                if self.args.insert:
                    # we only allow a insertion state can go back to original
                    # state or its child
                    if states[j].name == child_name:
                        insert_neighbor_prob = (1-self.args.insert_self_loop_epsilon) / 2
                        edges.append((insert_states[i], states[i], insert_neighbor_prob))
                        edges.append((insert_states[i], states[j], insert_neighbor_prob))

            if self.args.insert and child_name == model.end.name:
                insert_neighbor_prob = 1-self.args.insert_self_loop_epsilon
                edges.append((insert_states[i], states[i], insert_neighbor_prob))

            # each state can go to the end state
            edges.append((states[i], model.end, neighbor_prob))
            #if self.args.insert:
            #    edges.append((insert_states[i], model.end, neighbor_prob))
    
        states.extend([model.start, model.end])
        if self.args.insert:
            states.extend(insert_states)
        self.model = create_model(states, edges, model)
        self.state2child = state2child
        self.num_states = num_init_states
    

    def temperal_split(self, split_state):
        model_json = json.loads(self.model.to_json())
        named_edges, insert_named_edges = get_named_edges(model_json)
        state2emissionprob, insert_state2emissionprob, _ = get_states(model_json)
    
        child = self.state2child[split_state]
        new = f"T{self.num_temperal_split+1:02} <- {split_state}"
        if self.args.insert:
            insert_new = f'I@{new}'
    
        model = HiddenMarkovModel(name="Conversation HMM")
    
        # n2s: name2state
        if self.args.insert:
            global_emission_prob = {int(k): v for k, v in insert_state2emissionprob['I@S01'].items()}
            i_d = DiscreteDistribution(global_emission_prob)

        n2s = {}
        for name, ep in state2emissionprob.items(): 
            ep = {int(k): v for k, v in ep.items()}
            n2s[name] = State(DiscreteDistribution(ep), name=name)
            if self.args.insert:
                n2s[f'I@{name}'] = State(i_d, name=f'I@{name}')
            if name == split_state:
                n2s[new] = State(DiscreteDistribution(ep), name=new)
                if self.args.insert:
                    n2s[insert_new] = State(i_d, name=insert_new)
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
                elif j == f'I@{i}':
                    new_named_edges.append((n2s[new], n2s[insert_new], prob))
                    new_named_edges.append((n2s[i], n2s[j], prob))
                else:
                    new_named_edges.append((n2s[i], n2s[j], prob))
                    if i == j:
                        new_named_edges.append((n2s[new], n2s[new], prob))
                    else:
                        new_named_edges.append((n2s[new], n2s[j], prob))
            else:
                new_named_edges.append((n2s[i], n2s[j], prob))

        if self.args.insert:
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

        if self.args.topk_outgoing:
            new_named_edges = StateSplitingHMM.get_topk_outgoing(new_named_edges, model.end, self.args.topk_outgoing)
    
        model = create_model(list(n2s.values()), new_named_edges, model)
    
        return model, new


    def vertical_split(self, split_state):
    
        model_json = json.loads(self.model.to_json())
        named_edges, insert_named_edges = get_named_edges(model_json)
        state2emissionprob, insert_state2emissionprob, _ = get_states(model_json)
    
        new = f"V{self.num_vertical_split+1:02} <- {split_state}"
        if self.args.insert:
            insert_new = f'I@{new}'
    
        model = HiddenMarkovModel(name="Conversation HMM")
    
        # n2s: name2state
        if self.args.insert:
            global_emission_prob = {int(k): v for k, v in insert_state2emissionprob['I@S01'].items()}
            i_d = DiscreteDistribution(global_emission_prob)
        n2s = {}
        for name, ep in state2emissionprob.items(): 
            ep = {int(k): v for k, v in ep.items()}
            n2s[name] = State(DiscreteDistribution(ep), name=name)
            if self.args.insert:
                n2s[f'I@{name}'] = State(i_d, name=f'I@{name}')
            if name == split_state:
                max_prob_cluster = max(ep.items(), key=lambda x: x[1])[0]
                ep[max_prob_cluster] = 0
                #n2s[new] = State(DiscreteDistribution(ep), name=new)
                n2s[new] = State(DiscreteDistribution(self.global_emission_probs), name=new)
                if self.args.insert:
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

        if self.args.insert:
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

        if self.args.topk_outgoing:
            new_named_edges = StateSplitingHMM.get_topk_outgoing(new_named_edges, model.end, self.args.topk_outgoing)
    
        model = create_model(list(n2s.values()), new_named_edges, model)
    
        return model, new
    
    
    def plot(self, image_path):
        #df = pd.read_csv('./raw_data/150/medoid_centers.csv')
    
        model_json = json.loads(self.model.to_json())
    
        state2emissionprob, insert_state2emissionprob, dummy_states = get_states(model_json)
    
        g = Digraph('G', engine="dot")
    
        # match emission
        c0 = Digraph('cluster_0')
        c0.body.append('style=filled')
        c0.body.append('color=white')
        c0.attr('node', shape='box')
        c0.node_attr.update(color='#D0F0FB', style='filled',fontsize="14")
        #c0.edge_attr.update(color='#076789', fontsize="12")
        c0.edge_attr.update(color='white')

        if self.args.insert:
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
        sn2ep = list(state2emissionprob.items())
        if self.args.insert:
            sn2ep += list(insert_state2emissionprob.items())
        for name, ep in sn2ep:
            if name[0] == 'I':
                _emission_topk = self.args.plot_insert_topk_clusters
            else:
                _emission_topk = self.args.plot_topk_clusters

            string = name + "\n"
            topk = sorted(ep.items(), key=lambda x: x[1], reverse=True)[:_emission_topk]
            for cluster, prob in topk:
                string += f'cluster = {cluster}; prob = {prob:.4f}\l'
                string += "\l".join(textwrap.wrap(f"{self.cluster2utt[cluster]}", width=40))
                string += "\l\l"
            state2topk_clusters[name] = string
    
        """
            start and end
        """
        for s in dummy_states:
            c0.node(s)

        for j, _ in i2js[self.model.start.name]:
            c0.edge(self.model.start.name, j)

        """
            normal states
        """
        for name in state2emissionprob.keys():
            topk_clusters = state2topk_clusters[name] 
            c0.node(name, topk_clusters)
            # for better visualization, we tie child with parent by a white edge
            c0.edge(name, self.state2child[name])

        """
            insert states
        """
        if self.args.insert:
            for name in insert_state2emissionprob.keys():
                topk_clusters = state2topk_clusters[name]
                c1.node(name, topk_clusters)

                # for better visualization, we tie child with parent by a white edge
                normal_state_name = name.split('@')[1]
                child_name = self.state2child[normal_state_name]
                if child_name != self.model.end.name:
                    child_insert_state_name = f'I@{child_name}'
                    c1.edge(name, child_insert_state_name)

        g.subgraph(c0)
        if self.args.insert:
            g.subgraph(c1)

        # sort edges by their prob
        for i, js in i2js.items():
            i2js[i] = sorted(js, key=lambda x: x[1], reverse=True)

        for i in list(state2emissionprob.keys()) + list(insert_state2emissionprob.keys()) + [self.model.start.name]:
            red_top1 = False
            if i[0] == 'I':
                color = '#388B46'
            else:
                color = '#005FFE'
            js = i2js[i]
            for k, (j, prob) in enumerate(js):
                if j == self.model.end.name:
                    g.edge(i, j, label=f'{prob:.3f}', len='3.00', style='dashed', color=color)
                elif i != j and not red_top1 and i[0] != 'I':
                    g.edge(i, j, label=f'{prob:.3f}', len='3.00', color='#F0644D')
                    red_top1 = True
                else:
                    g.edge(i, j, label=f'{prob:.3f}', len='3.00', color=color)
    
        # edge
        #idx2names = {i: s['name'] for i, s in enumerate(model_json['states'])}
        #edges = []
        #for e in model_json['edges']:
        #    i = idx2names[e[0]]
        #    j = idx2names[e[1]]
        #    prob = e[2]
        #    if j == self.model.end.name:
        #        c0.edge(i, j, label=f'{prob:.3f}', len='3.00', style='dashed')
        #    else:
        #        c0.edge(i, j, label=f"{prob:.3f}", len='3.00')
    
        ##the graph is basicaly split in 4 clusters
        #g.subgraph(c0)
    
        g.edge_attr.update(arrowsize='1')
        g.body.extend(['rankdir=LR', 'size="160,100"'])
        g.render(image_path, format='pdf')


    @staticmethod
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
