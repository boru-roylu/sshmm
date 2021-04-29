#!/bin/bash
source /g/ssli/transitory/heddayam/miniconda3/etc/profile.d/conda.sh
conda activate hmm
cd /g/ssli/transitory/heddayam/sshmm
python sshmm.py --num_split 8 --seq_data /g/ssli/data/tmcalls/heddayam/gridspace-stanford-harper-valley/sshmm --cluster_representative_path /g/ssli/data/tmcalls/heddayam/gridspace-stanford-harper-valley/clustering/kmedoids-agent-30-centers.csv
#python sshmm.py --num_split 12 --seq_data /g/ssli/data/tmcalls/sshmm --cluster_representative_path /g/ssli/data/tmcalls/sshmm/kmedoids_150_centers_labels_v1.csv
