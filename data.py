import os
import torch
import numpy as np


def get_datasets(parent_dir):
    lines = []

    vocab = set()
    data = {"train": [], "dev": []}
    for split in ["train", "dev"]:
        path = os.path.join(parent_dir, f"{split}.txt")

        with open(path, 'r') as f:
            for l in f:
                l = [int(ll) for ll in l.strip().split(',')]
                data[split].append(l)
                vocab.update(set(l))

    vocab = {v: i for i, v in enumerate(vocab)}

    # get input and output alphabets
    train_dataset = TextDataset(data["train"], vocab)
    dev_dataset = TextDataset(data["dev"], vocab)

    return train_dataset, dev_dataset, vocab


#class TextDataset(torch.utils.data.Dataset):
class TextDataset:
    def __init__(self, data, vocab):
        self.data= data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = list(map(self.vocab.get, self.data[idx]))

        return np.array(x)


class PadAndOneHot:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, batch):
        """
        Returns a minibatch of strings, one-hot encoded and padded to have the same length.
        """
        xs = []
        x_lens = []
        batch_size = len(batch)
        for i in range(batch_size):
            x = batch[i]
            x = [self.vocab[xx] for xx in x]
            xs.append(x)
            x_lens.append(len(x))

        # pad all sequences with 0 to have same length
        T = max(x_lens)
        for i in range(batch_size):
            xs[i] += [0] * (T - x_lens[i])
            xs[i] = torch.tensor(xs[i])

        xs = torch.stack(xs)
        x_lens = torch.tensor(x_lens)

        return (xs, x_lens)
