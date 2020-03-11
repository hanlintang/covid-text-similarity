import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from transformers.optimization import AdamW
from utils import init_arg_parser
from data import Text


def init_configs():
    arg_parser = init_arg_parser()
    args = arg_parser.parse_args()
    return args


def train(args):
    """
    训练
    """
    # data preprocess
    train_data = Text('./data/train/train_20200228.csv')
    dataloader = DataLoader(train_data, shuffle=True, batch_size=32)

    # model defination
    model = BertForSequenceClassification.from_pretrained(
        './pretrained', num_labels=2)
    model.to(args.device)
    model.train()
    # dev(model)

    # optimizer
    param_optimizer = list(model.named_parameters())
    # print(param_optimizer[0][0])
    # TODO: figure out why these parameters need no decay
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {
            'params':
                [p for n, p in param_optimizer
                 if not any(nd in n for nd in no_decay)],
            'weight_decay':
                0.01
        },
        {
            'params':
                [p for n, p in param_optimizer
                 if any(nd in n for nd in no_decay)],
            'weight_decay':
                0.0
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-05)
    # training
    for i in range(10):
        for step, batch_data in enumerate(dataloader):
            batch_data = tuple(t.to(args.device) for t in batch_data)
            # print(len(batch_data[0]))
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels =\
                batch_data
            logits = model(
                batch_seqs, batch_seq_masks, batch_seq_segments, labels=None)
            # TODO: is there any difference between the two positions
            # of the loss defination
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(logits[0], batch_labels)
            optimizer.zero_grad()
            loss.backward()
            print("epoch{}, step {}, loss = {}".format(i, step, loss.item()))
            optimizer.step()
        # save checkpoints
        save_to = './checkpoints/'
        if args.save_models:
            model_file = save_to + '.iter%d.bin' % i
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save_pretrained('./checkpoints')
    # dev stage
    dev(args, model)


def dev(args, model):
    """
    调参
    """
    dev_dataset = Text('./data/dev/dev_20200228.csv')
    dataloader = DataLoader(dev_dataset, shuffle=True, batch_size=32)

    true_labels = []
    pred_labels = []
    model.eval()

    with torch.no_grad():
        for batch_data in dataloader:
            batch_data = tuple(t.to(args.device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels =\
                batch_data
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


def test(args):
    """
    测试
    """
    dev_dataset = Text('./data/dev/dev_20200228.csv')
    dataloader = DataLoader(dev_dataset, shuffle=True, batch_size=32)
    model = BertForSequenceClassification.from_pretrained(
        './checkpoints/', num_labels=2)
    model.to(args.device)
    true_labels = []
    pred_labels = []
    model.eval()
    # print('fine')
    with torch.no_grad():
        for batch_data in dataloader:
            batch_data = tuple(t.to(args.device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels =\
                batch_data
            logits = model(
                batch_seqs, batch_seq_masks, batch_seq_segments, labels=None)
            # print(logits)
            # print(logits[0].shape)
            logits = logits[0].argmax(dim=1)
            # print(logits)
            pred_labels += logits.cpu().numpy().tolist()
            true_labels += batch_labels.cpu().numpy().tolist()

    acc_cnt = 0
    for l_pre, l_true in zip(pred_labels, true_labels):
        if l_pre == l_true:
            acc_cnt += 1
    print('valid acc: {}'.format(acc_cnt / len(pred_labels)))
    # 查看各个类别的准召


def help():
    """
    打印帮助信息
    """
    print('TODO')


if __name__ == "__main__":
    args = init_configs()
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
