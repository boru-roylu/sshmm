import os
import sys
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_parent_dir",
    required=True,
    type=str,
    help="",
)
parser.add_argument(
    "--sep",
    default=",",
    type=str,
    help="",
)
parser.add_argument(
    "--party",
    default="agent",
    type=str,
    help="",
)
args = parser.parse_args()

for split in ['train', 'dev', 'test']:
    df_path = f'{args.data_parent_dir}/{args.party}_{split}.csv'
    df = pd.read_csv(df_path, sep=args.sep)
    data = []
    groups = df.groupby(df.example_id)
    for i, (_, utts) in enumerate(groups):
        if i % 100 == 0:
            print(f'progress: {i / len(groups) * 100:.1f}%', end='\r')
        sid = utts["example_id"].tolist()[0]
        seq = utts["cluster"].astype(str).tolist()

        data.append((sid, ",".join(seq)))

    df = pd.DataFrame(data, columns=["example_id", "cluster_sequence"])

    output_dir = f'{args.data_parent_dir}/cluster_sequence'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{args.party}_{split}.csv')
    df.to_csv(output_path, sep="|", index=False)
    print(f'[{split}] # of example {len(df)}')
