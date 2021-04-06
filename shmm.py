import os
import sys
import json
import numpy as np

from data2 import get_datasets

from pomegranate import *

from utils import save_model, get_states, normalize, entropy, get_named_edges
from shmm_utils import init_model, plot_shmm, temperal_split

if __name__ == '__main__':
    num_clusters = int(sys.argv[1])
    num_init_states = 3
    num_split = 12
    max_iterations = 10

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
    plot_shmm(model, image_path=os.path.join(image_dir, f'shmm_init'))

    num_temperal_split = 0
    print('Start training')
    print(f'********** iteration 0 **********')
    print(f'num_states = {num_states}; num_temperal_split = {num_temperal_split}')
    model.fit(xs, algorithm='baum-welch', emission_pseudocount=1, stop_threshold=0.001, max_iterations=max_iterations, verbose=True, n_jobs=os.cpu_count()-2)
    plot_shmm(model, image_path=os.path.join(image_dir, f'shmm_{num_states:02}'))
    for iteration in range(num_split):

        print(f'********** iteration {iteration+1} **********')
        model_json = json.loads(model.to_json())
        state2emissionprob, _ = get_states(model_json)
        max_entropy_state = max(state2emissionprob.items(), key=lambda x: entropy(list(x[1].values())))[0]
        model, state2child = temperal_split(model_json, max_entropy_state, state2child, num_temperal_split)

        num_states += 1
        num_temperal_split += 1
        print(f'num_states = {num_states}; num_temperal_split = {num_temperal_split}')

        model.fit(xs, algorithm='baum-welch', stop_threshold=0.001, max_iterations=max_iterations, verbose=True, n_jobs=os.cpu_count()-2)
        plot_shmm(model, image_path=os.path.join(image_dir, f'shmm_{num_states:02}'))
        save_model(num_states, model, exp_dir)

    exit()
