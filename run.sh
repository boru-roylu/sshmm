python sshmm.py \
  --num_clusters 50 \
  --max_iter 13 \
  --seq_data ./data/kmedoids_agent_150_merge_num \
  --center ./raw_data/kmedoids_agent_150_merge_num/manual_centers.csv \
  --exp ./exp/shmm_tv_merge_num_manual_insert2 \
  --manual \
  $@
