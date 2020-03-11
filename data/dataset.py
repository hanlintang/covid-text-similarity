# import os
from torch.utils import data
import numpy as np
import pandas as pd
from transformers import BertTokenizer
import torch


MAX_SENT_LENGTH = 100
VOCAB_PATH = './pretrained/vocab.txt'


class Text(data.Dataset):
    def __init__(self, filepath):
        """
        整理数据集
        """
        pair, label = readfile(filepath)

        data = {
            'sent': [],
            'sent_mask': [],
            'sent_segment': []
        }

        tokenizer = BertTokenizer.from_pretrained(VOCAB_PATH)

        for sent1, sent2 in pair:
            sent = '[CLS]' + sent1 + ' [SEP] ' + sent2 + ' [SEP]'
            token_list = tokenizer.tokenize(sent)
            # find the sep position
            for i, word in enumerate(token_list):
                if word == '[SEP]':
                    sent1_len = i + 1
                    break

            sent_id = tokenizer.convert_tokens_to_ids(token_list)
            padding_id = [0] * (MAX_SENT_LENGTH - len(token_list))
            data['sent'].append(sent_id + padding_id)

            assert len(data['sent'][-1]) == 100
            # print(len(data['sent'][-1]))
            data['sent_segment'].append(
                [1] * (sent1_len) + [0] * (MAX_SENT_LENGTH - sent1_len))
            # TODO: 暂时还没搞懂mask加在哪
            data['sent_mask'].append([1] * len(token_list) + padding_id)

        self.t_seqs = torch.tensor(data['sent'], dtype=torch.long)
        self.t_seq_segs = torch.tensor(data['sent_segment'], dtype=torch.long)
        self.t_seq_masks = torch.tensor(data['sent_mask'], dtype=torch.long)
        self.t_labels = torch.tensor(label, dtype=torch.long)
        # print(self.t_labels)

    def __getitem__(self, index):
        return (self.t_seqs[index],
                self.t_seq_segs[index],
                self.t_seq_masks[index],
                self.t_labels[index])

    def __len__(self):
        return len(self.t_labels)


def readfile(filepath):
    """
    读取数据
    """
    csv_data = pd.read_csv(filepath)
    sent_pair, label = [], []
    for i in range(len(csv_data)):
        line = csv_data.iloc[i].tolist()
        if np.isnan(line[4]):
            continue
        sent_pair.append([line[2], line[3]])
        label.append(line[4])

    return sent_pair, label


if __name__ == "__main__":
    train_data = Text('./data/train/train_20200228.csv')
    for a, b, c, d in train_data:
        print(a)
        print(b)
        break
