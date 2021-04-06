import os
import json
import numpy as np


from data2 import get_datasets

from pomegranate import *

from utils import save_model, get_states, normalize, entropy, get_named_edges
from shmm_utils import init_model, plot_shmm, temperal_split

if __name__ == '__main__':
    num_clusters = 50
    num_init_states = 3
    num_split = 12
    exp_dir = f"./exp/shmm/{num_clusters}"
    image_dir = f"./images/shmm/{num_clusters}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    train_dataset, dev_dataset, vocab, cnt = get_datasets("./data/kmedoids_agent_150", num_clusters)
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

    num_states = num_init_states
    model, state2child = init_model(xs, num_init_states, cnt, init_threshold=init_threshold)

    model_json = json.loads(model.to_json())
    state2emissionprob, _ = get_states(model_json)

    max_entropy_state = max(state2emissionprob.items(), key=lambda x: entropy(list(x[1].values())))[0]
    plot_shmm(model, image_path=os.path.join(image_dir, f'init_shmm'))

    num_temperal_split = 0
    print('Start training')
    print(f'********** iteration 0 **********')
    print(f'num_states = {num_states}; num_temperal_split = {num_temperal_split}')
    model.fit(xs, algorithm='baum-welch', emission_pseudocount=1, stop_threshold=0.001, max_iterations=15, verbose=True, n_jobs=os.cpu_count()-2)
    plot_shmm(model, image_path=os.path.join(image_dir, f'shmm_{num_states}'))

    for iteration in range(num_split):
        print(f'********** iteration {iteration+1} **********')
        model, state2child = temperal_split(model_json, max_entropy_state, state2child, num_temperal_split)

        num_states += 1
        num_temperal_split += 1
        print(f'num_states = {num_states}; num_temperal_split = {num_temperal_split}')

        model.fit(xs, algorithm='baum-welch', emission_pseudocount=1, stop_threshold=0.001, max_iterations=15, verbose=True, n_jobs=os.cpu_count()-2)
        plot_shmm(model, image_path=os.path.join(image_dir, f'shmm_{num_states}'))
        model.save_model(num_states, model, exp_dir)

    exit()

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
