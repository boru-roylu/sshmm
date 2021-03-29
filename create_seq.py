import pandas as pd

n_clusters = 50
#train = f"./exp/kmeans_agent_{n_clusters}_clusters/train.csv"
#dev = f"./exp/kmeans_agent_{n_clusters}_clusters/dev.csv"
train = f"./raw_data/150/kmedoids_agent_train_clusters.csv.gz"
dev = f"./raw_data/150/kmedoids_agent_dev_clusters.csv.gz"



for split, csv in [("train", train), ("dev", dev)]: 
    df = pd.read_csv(csv, compression="gzip")
    df = df[df.party == "agent"]
    data = []
    no_good_start = 0
    for _, utts in df.groupby(df.sourcemediaid):
        #utts = utts.sort_values(by="idx")
        sid = utts["sourcemediaid"].tolist()[0]
        seq = utts["cluster"].astype(str).tolist()

        #if seq[0] != "45":
        #    no_good_start += 1
        #    continue

        data.append((sid, ",".join(seq)))

    df = pd.DataFrame(data, columns=["sourcemediaid", "cluster_sequence"])
    df.to_csv(f"{split}.csv", sep="|", index=False)
    print(len(df))
    print(no_good_start)
