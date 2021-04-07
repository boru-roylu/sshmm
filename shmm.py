import os
import sys
import json
import numpy as np

from data2 import get_datasets

from pomegranate import *

from utils import save_model, get_states, normalize, entropy, get_named_edges
from shmm_utils import init_model, plot_shmm, temporal_split, vertical_split


num_clusters = int(sys.argv[1])
num_init_states = 3
num_split = 12
max_iterations = 10

exp_dir = f"./exp/shmm_debug/{num_clusters}"
os.makedirs(exp_dir, exist_ok=True)
image_dir = f"./images/shmm_debug/{num_clusters}"
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
plot_shmm(model, state2child, image_path=os.path.join(image_dir, f'shmm_init'))

num_temporal_split = 0
num_vertical_split = 0
print('Start training')
print(f'********** iteration 0 **********')
print(f'num_states = {num_states}; num_temporal_split = {num_temporal_split}; num_vertical_split = {num_vertical_split}')

model.fit(xs, algorithm='baum-welch', emission_pseudocount=1, stop_threshold=20,
          max_iterations=max_iterations, verbose=True, n_jobs=os.cpu_count()-2)

plot_shmm(model, state2child, image_path=os.path.join(image_dir, f'shmm_{num_states:02}'))

for iteration in range(num_split):

    print('*'*20, f'iteration {iteration+1}', '*'*20)


    model_json = json.loads(model.to_json())
    state2emissionprob, _, _ = get_states(model_json)
    max_entropy_state = max(state2emissionprob.items(), key=lambda x: entropy(list(x[1].values())))[0]
    t_model, t_state2child = temporal_split(model_json, max_entropy_state, state2child, num_temporal_split, topk_outgoing=5)
    v_model, v_state2child = vertical_split(model_json, max_entropy_state, state2child, num_vertical_split, topk_outgoing=5)

    print("    Try temporal split")
    t_model.fit(xs, algorithm='baum-welch', emission_pseudocount=1, stop_threshold=20,
              max_iterations=max_iterations, verbose=True, n_jobs=os.cpu_count()-2)
    print()
    print("    Try vertical split")
    v_model.fit(xs, algorithm='baum-welch', emission_pseudocount=1, stop_threshold=20,
              max_iterations=max_iterations, verbose=True, n_jobs=os.cpu_count()-2)
    print()

    t_logprob = sum(t_model.log_probability(x) for x in xs)
    v_logprob = sum(v_model.log_probability(x) for x in xs)

    print(f"    LogProb: temporal = {t_logprob:.3f}; vertical = {v_logprob:.3f}")

    if t_logprob + 1e5 > v_logprob:
        print("    Choose temporal split")
        model = t_model
        state2child = t_state2child
        num_temporal_split += 1
    else:
        print("    Choose vertical split")
        model = v_model
        state2child = v_state2child
        num_vertical_split += 1
    num_states += 1

    print(f'    Info: num_states = {num_states}; num_temporal_split = {num_temporal_split}; num_vertical_split = {num_vertical_split}')

    plot_shmm(model, state2child, image_path=os.path.join(image_dir, f'shmm_{num_states:02}'))
    save_model(num_states, model, exp_dir)
