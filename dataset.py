import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os

punctuation_token2symbol = {
    "[PERIOD]": [".", "!", ";"],
    "[COMMA]": [",", ":", '"'],
    "[QUESTION]": ["?"],
}

punctuation_token2id = {
    "[EMPTY]": 0,
    "[PERIOD]": 1,
    "[COMMA]": 2,
    "[QUESTION]": 3,
}

punctuation_id2token = {
    0: "[EMPTY]",
    1: "[PERIOD]",
    2: "[COMMA]",
    3: "[QUESTION]",
}


def parse_data(file_path, tokenizer):
    df = pd.read_csv(file_path)

    x = []
    y = []
    y_mask = []
    attn_mask = []

    for i in range(len(df)):
        word, punkt = df.iloc[i]
        try:
            tokens = tokenizer.tokenize(word)
        except:
            pass

        if len(tokens) == 0:
            x = x + tokenizer.encode("[UNK]", add_special_tokens=False)
        else:
            x = x + tokenizer.convert_tokens_to_ids(tokens)
            y = y + [punctuation_token2id["[EMPTY]"]] * (len(tokens) - 1)
            y_mask = y_mask + [0] * (len(tokens) - 1)
            attn_mask = attn_mask + [1] * (len(tokens) - 1)

        y.append(punctuation_token2id[punkt])
        y_mask.append(1)
        attn_mask.append(1)

    return x, y, y_mask, attn_mask


class CorpusDataset(Dataset):
    def __init__(
        self,
        dir,
        tokenizer,
        sequence_len=256,
        transform=None,
        target_transform=None,
        documents_limit=200,
    ):
        self.dir = dir
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len - 2
        self.transform = transform
        self.target_tranform = target_transform

        self.x_list = []
        self.y_list = []
        self.y_mask_list = []
        self.attn_mask_list = []
        self.lengths = []

        for file_name in os.listdir(self.dir)[:documents_limit]:
            file_path = os.path.join(self.dir, file_name)
            x, y, y_mask, attn_mask = parse_data(file_path, self.tokenizer)
            self.x_list.append(x)
            self.y_list.append(y)
            self.y_mask_list.append(y_mask)
            self.attn_mask_list.append(attn_mask)
            self.lengths.append(max(0, len(x) - self.sequence_len + 1))

        self.lengths = np.asarray(self.lengths)
        self.lengths_cumsum = self.lengths.cumsum()

        self.cls_token = tokenizer.encode("[CLS]", add_special_tokens=False)
        self.sep_token = tokenizer.encode("[SEP]", add_special_tokens=False)

    def __len__(self):
        return self.lengths_cumsum[-1]

    def __getitem__(self, idx):
        document_idx = np.where(self.lengths_cumsum - 1 >= idx)[0][0]
        if document_idx == 0:
            idx_start = idx
        else:
            idx_start = int(idx - self.lengths_cumsum[document_idx - 1])

        x = self.x_list[document_idx]
        y = self.y_list[document_idx]
        y_mask = self.y_mask_list[document_idx]
        attn_mask = self.attn_mask_list[document_idx]
        x = torch.tensor(
            self.cls_token
            + x[idx_start : idx_start + self.sequence_len]
            + self.sep_token
        )

        y = torch.tensor(
            [punctuation_token2id["[EMPTY]"]]
            + y[idx_start : idx_start + self.sequence_len]
            + [punctuation_token2id["[EMPTY]"]]
        )

        attn_mask = torch.tensor(
            [1] + attn_mask[idx_start : idx_start + self.sequence_len] + [1]
        )

        y_mask = torch.tensor(
            [0] + y_mask[idx_start : idx_start + self.sequence_len] + [0]
        )

        return x, y, attn_mask, y_mask
