import pandas as pd

train = "./raw_data/kmedoids_both_train_clusters.csv.gz"
dev = "./raw_data/kmedoids_both_dev_clusters.csv.gz"



for split, csv in [("train", train), ("dev", dev)]: 
    df = pd.read_csv(csv, compression="gzip")
    df = df[df.party == "agent"]
    data = []
    no_good_start = 0
    for _, utts in df.groupby(df.sourcemediaid):
        utts = utts.sort_values(by="idx")
        sid = utts["sourcemediaid"].tolist()[0]
        seq = utts["cluster"].astype(str).tolist()

        if seq[0] != "360":
            no_good_start += 1
            continue

        data.append((sid, ",".join(seq)))

    df = pd.DataFrame(data, columns=["sourcemediaid", "cluster_sequence"])
    df.to_csv(f"{split}.csv", sep="|", index=False)
