import json
import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                en, fr = line.strip().split("\t")
                en = torch.tensor(json.loads(en))
                fr = torch.tensor(json.loads(fr))
                self.data.append((en, fr))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def pad_batch(batch):
    en_batch, fr_batch = zip(*batch)

    en_padded = torch.nn.utils.rnn.pad_sequence(
        en_batch, batch_first=True, padding_value=0
    )
    fr_padded = torch.nn.utils.rnn.pad_sequence(
        fr_batch, batch_first=True, padding_value=0
    )

    return en_padded, fr_padded
