import os
import numpy as np

from collections import Counter

from data2 import get_datasets

from pomegranate import *

from utils import plotHMM

def normalize(dic):
    _sum = sum(dic.values())
    return {k: v/_sum for k, v in dic.items()}

def init_model(xs, num_m_states, count, init_threshold):
    model = HiddenMarkovModel(name="Conversation HMM")

    # Define the distribution for insertion states
    global_emission_probs = normalize(count)

    # Define the distribution for match states
    segment_ratios = [0.05]
    segment_ratios += [0.90/(num_m_states-2)]*(num_m_states-2)
    segment_ratios += [0.05]
    segment_ratios = np.array(segment_ratios)
    segment_cnts = [Counter() for _ in range(num_m_states)]
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

    m_states = []
    d_states = []
    i_d = DiscreteDistribution(global_emission_probs)
    i_states = [State(i_d, name=f"I00")]
    for k in range(num_m_states):

        #if k % 5 == 0 and k != 0:
        #    i_d = DiscreteDistribution(global_emission_probs)

        i = State(i_d, name=f"I{k+1:02}")
        i_states.append(i)

        m = State(DiscreteDistribution(emissionprobs[k]), name=f"M{k+1:02}")
        m_states.append(m)

        d = State(None, name=f"D{k+1:02}")
        d_states.append(d)

    model.add_states(i_states + m_states + d_states)

    # Create transitions from match states
    model.add_transition(model.start, m_states[0], 0.9)
    model.add_transition(model.start, i_states[0], 0.05)
    model.add_transition(model.start, d_states[0], 0.05)

    for i in range(num_m_states-1):
        model.add_transition(m_states[i], m_states[i+1], 0.9)
        model.add_transition(m_states[i], i_states[i+1], 0.05)
        model.add_transition(m_states[i], d_states[i+1], 0.05)

    model.add_transition(m_states[-1], model.end, 0.9)
    model.add_transition(m_states[-1], i_states[-1], 0.1)


    # Create transitions from insert states
    for i in range(num_m_states):
        model.add_transition(i_states[i], i_states[i], 0.70)
        model.add_transition(i_states[i], d_states[i], 0.15)
        model.add_transition(i_states[i], m_states[i], 0.15)
    model.add_transition(i_states[-1], i_states[-1], 0.85)
    model.add_transition(i_states[-1], model.end, 0.15)


    # Create transitions from delete states
    for i in range(num_m_states-1):
        model.add_transition(d_states[i], d_states[i+1], 0.15)
        model.add_transition(d_states[i], i_states[i+1], 0.15) 
        model.add_transition(d_states[i], m_states[i+1], 0.70)

    model.add_transition(d_states[-1], i_states[-1], 0.3)
    model.add_transition(d_states[-1], model.end, 0.7)

    # Call bake to finalize the structure of the model.
    model.bake()

    return model


def path_to_alignment(x, y, path):
    """
    This function will take in two sequences, and the ML path which is their alignment,
    and insert dashes appropriately to make them appear aligned. This consists only of
    adding a dash to the model sequence for every insert in the path appropriately, and
    a dash in the observed sequence for every delete in the path appropriately.
    """

    for i, (index, state) in enumerate(path[1:-1]):
        name = state.name

        if name.startswith('D'):
            y = y[:i] + [-1] + y[i:]
        elif name.startswith('I'):
            x = x[:i] + [-1] + x[i:]

    return x, y

def ints2str(ints, mapping=None):
    string = ''
    for i in ints:
        if mapping:
            string += f" {mapping[i]}"
        else:
            string += f" {i:>3}"
    return string


if __name__ == '__main__':
    num_clusters = 100
    num_match_states = 6

    train_dataset, dev_dataset, vocab, cnt = get_datasets("./data/kmedoids_agent_150", 100)
    print('vocab size = ', len(vocab))

    xs = []
    ids = []
    for x, _id in train_dataset:
        xs.append(x)
        ids.append(_id)
    xs = xs
    ids = ids
    x_lens = [len(x) for x in xs]

    init_threshold = int(np.mean(x_lens))

    model = init_model(xs, num_match_states, cnt, init_threshold=init_threshold)

    print('Start training')
    model.fit(xs, algorithm='baum-welch',stop_threshold=0.001, max_iterations=20, verbose=True, n_jobs=os.cpu_count()-2)

    plotHMM(model)

    for x in xs[:100]:
        logp, path = model.viterbi(x)
        x = x.tolist()
        print(f"Log Probability: {logp}")
        x_str = ' '.join(f"{xx:>3}" for xx in x)
        print(f"Sequence: {x_str}")
        s_str = ' '.join(f"{state.name:>3}" for _, state in path[1:-1])
        print(f"Path:     {s_str}")
        print()

    mapping = {k: chr(i) for i, k in enumerate(cnt.keys(), start=200)}
    mapping[-1] = '-'
    for x in xs[1:]:
        logp, path = model.viterbi(x)
        x2, y = path_to_alignment(xs[0].tolist(), x.tolist(), path)

        x = ints2str(x.tolist())
        x2 = ints2str(x2)#, mapping)
        y = ints2str(y)#, mapping)
        print(f"Log Probability: {logp}")
        print(f"Sequence:")
        print(x)
        print(x2)
        print()
        print(y)
        print()
