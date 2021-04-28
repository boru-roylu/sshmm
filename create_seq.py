import os
import sys
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_path",
    required=True,
    type=str,
    help="",
)
parser.add_argument(
    "--dev_path",
    required=True,
    type=str,
    help="",
)
parser.add_argument(
    "--output_parent_dir",
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
args = parser.parse_args()

os.makedirs(args.output_parent_dir, exist_ok=True)

for split, csv in [("train", args.train_path), ("dev", args.dev_path)]: 
    df = pd.read_csv(csv, sep=args.sep, compression="gzip")
    df = df[df.party == "agent"]
    data = []
    for _, utts in df.groupby(df.sourcemediaid):
        sid = utts["sourcemediaid"].tolist()[0]
        seq = utts["cluster"].astype(str).tolist()

        data.append((sid, ",".join(seq)))

    df = pd.DataFrame(data, columns=["sourcemediaid", "cluster_sequence"])

    output_path = os.path.join(args.output_parent_dir, f"{split}.csv")
    df.to_csv(output_path, sep="|", index=False)
    print(f'[{split}] # of example {len(df)}')
