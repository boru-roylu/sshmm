#!/bin/bash
source /g/ssli/transitory/heddayam/miniconda3/etc/profile.d/conda.sh
conda activate hmm
cd /g/ssli/transitory/heddayam/sshmm
python run.py --topk_cluster 30 --targeted_num_states 15
