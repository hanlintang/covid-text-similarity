from transformers import BertTokenizer, BertForSequenceClassification
from transformers.optimization import AdamW
import pandas as pd
import numpy as np
import torch
import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader

def read_data(filepath):
    csv_data = pd.read_csv(filepath)
    sent_pair, label = [], []
    for i in range(len(csv_data)):
        line = csv_data.iloc[i].tolist()
        if np.isnan(line[3]):
            continue
        sent_pair.append([line[1], line[2]])
        label.append(line[3])
    return sent_pair, label

MAX_Q_LENGTH = 100

def main():

    tokenizer = BertTokenizer.from_pretrained('bert-chinese')

    train_sent_pair, train_label = read_data('./data/train/train.csv')
    train_data = {
        'sent': [],
        'sent_mask':[],
        'sent_segment':[]
    }

    for q1, q2 in train_sent_pair:
        sent = '[CLS]' + q1 + '[SEP]' + q2 + '[SEP]'
        token_list = tokenizer.tokenize(sent)
        for i, word in enumerate(token_list):
            if word == '[SEP]':
                q1_len = i + 1
                break
        sent_id = tokenizer.convert_tokens_to_ids(token_list)
        padding_id = [0] * (MAX_Q_LENGTH - len(token_list))
        train_data['sent'].append(sent_id + padding_id)
        train_data['sent_segment'].append([1] * (q1_len) + [0] * (MAX_Q_LENGTH - q1_len))
        train_data['sent_mask'].append([1] * len(token_list) + padding_id)

    t_seqs = torch.tensor(train_data['sent'], dtype=torch.long)
    t_seq_segs = torch.tensor(train_data['sent_segment'], dtype=torch.long)
    t_seq_masks = torch.tensor(train_data['sent_mask'], dtype=torch.long)
    t_labels = torch.tensor(train_label, dtype=torch.long)

    dataset = TensorDataset(t_seqs, t_seq_masks, t_seq_segs, t_labels)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=32)

    device = "cpu"  #    'cuda:0'

    model = BertForSequenceClassification.from_pretrained('bert-chinese', num_labels=2)
    model.to(device)
    model.train()

    param_optimizer = list(model.named_parameters())
    #print(param_optimizer)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {
            'params':
                [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay':
                0.01
        },
        {
            'params':
                [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
                0.0
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                         lr=2e-05)

    for i in range(10):
        for step, batch_data in enumerate(
                dataloader):
            batch_data = tuple(t.to(device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = batch_data
            logits = model(
                batch_seqs, batch_seq_masks, batch_seq_segments, labels=None)
            loss_function = CrossEntropyLoss()
            loss = loss_function(logits[0], batch_labels)
            optimizer.zero_grad()
            loss.backward()
            print("epoch{}, step {}, loss = {}".format(i, step, loss.item()))
            optimizer.step()


    dev_sent_pair, dev_label = read_data('./dev.csv')
    dev_data = {
        'sent': [],
        'sent_mask': [],
        'sent_segment': []
    }

    for q1, q2 in dev_sent_pair:
        sent = '[CLS]' + q1 + '[SEP]' + q2 + '[SEP]'
        token_list = tokenizer.tokenize(sent)
        for i, word in enumerate(token_list):
            if word == '[SEP]':
                q1_len = i + 1
                break
        sent_id = tokenizer.convert_tokens_to_ids(token_list)
        # print(len(token_list) == len(sent_id))
        padding_id = [0] * (MAX_Q_LENGTH - len(token_list))
        dev_data['sent'].append(sent_id + padding_id)
        dev_data['sent_segment'].append([1] * (q1_len) + [0] * (MAX_Q_LENGTH - q1_len))
        dev_data['sent_mask'].append([1] * len(token_list) + padding_id)

    t_seqs = torch.tensor(dev_data['sent'], dtype=torch.long)
    t_seq_segs = torch.tensor(dev_data['sent_segment'], dtype=torch.long)
    t_seq_masks = torch.tensor(dev_data['sent_mask'], dtype=torch.long)
    t_labels = torch.tensor(dev_label, dtype=torch.long)

    dataset = TensorDataset(t_seqs, t_seq_masks, t_seq_segs, t_labels)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=32)

    true_labels = []
    pred_labels = []
    model.eval()

    with torch.no_grad():
        for batch_data in dataloader:
            batch_data = tuple(t.to(device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = batch_data
            logits = model(
                batch_seqs, batch_seq_masks, batch_seq_segments, labels=None)
            logits = logits[0].argmax(dim=1)
            pred_labels += logits.cpu().numpy().tolist()
            true_labels += batch_labels.cpu().numpy().tolist()

    acc_cnt = 0
    for l_pre, l_true in zip(pred_labels, true_labels):
        if l_pre == l_true:
            acc_cnt += 1
    print('valid acc: {}'.format(acc_cnt / len(pred_labels)))
    # 查看各个类别的准召
if __name__ == '__main__':
    main()