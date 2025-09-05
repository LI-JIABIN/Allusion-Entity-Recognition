# -*- encoding: utf-8 -*-
import json
#pip install scikit-learn
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import argparse
from torch.utils import data
from crf import Bert_BiLSTM_CRF
from utils import NerDataset, pad, VOCAB, tokenizer, tag2idx, idx2tag
from config import Logger
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse(tokens, pred_tags):
    """Parse the predict tags to real word

    Arguments:
        x {List[str]} -- the origin text
        pred_tags {List[str]} -- predicted tags

    Return:
        entities {List[str]} -- a list of entities
    """
    entities = []
    entity = None
    # print(pred_tags)
    for idx, st in enumerate(pred_tags):
        # print(idx, st)
        if entity is None:
            if st.startswith('B'):
                entity = {}
                entity['start'] = idx
            else:
                continue
        else:
            if st == 'S':
                entity['end'] = idx
                name = ''.join(tokens[entity['start']: entity['end']])
                entities.append(name)
                entity = None
            elif st.startswith('B'):
                entity['end'] = idx
                name = ''.join(tokens[entity['start']: entity['end']])
                entities.append(name)
                entity = {}
                entity['start'] = idx
            else:
                continue
    return entities








def train(model, iterator, optimizer, device):
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens = batch
        # print("Words:", words)
        # print("Input IDs:", x)
        # print("Tags:", tags)
        # print("Label IDs:", y)
        # print('*'*100)
        x = x.to(device)
        y = y.to(device)
        _y = y # for monitoring
        optimizer.zero_grad()
        loss = model.neg_log_likelihood(x, y) # logits: (N, T, VOCAB), y: (N, T)

        loss.backward()

        optimizer.step()

        if i%10==0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")



def eval(model, iterator, device):
    #单独评价时
    # BEST_MODEL_PATH = r'./checkpoints/best_model.pt'
    # model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch
            x = x.to(device)

            _, y_hat = model(x)  # y_hat: (N, T)

            # 将 batch 的数据记录下来
            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    # 将预测结果转换为实际标签
    all_true_tags = []
    all_pred_tags = []

    for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
        # print(words)
        # words 和 tags 已经是列表，不需要调用 split()
        words = words.split()
        tags = tags.split()

        y_hat = [idx2tag[hat] for hat in y_hat]

        # 将预测标签与真实标签对齐，只取 is_heads 为 True 的位置
        true_tags = [tag for head, tag in zip(is_heads, tags) if head]
        pred_tags = [tag for head, tag in zip(is_heads, y_hat) if head]
        print(true_tags)
        print(pred_tags)
        all_true_tags.extend(true_tags)
        all_pred_tags.extend(pred_tags)

    # 计算四个指标
    precision = precision_score(all_true_tags, all_pred_tags, average='macro')  #, average='micro'
    recall = recall_score(all_true_tags, all_pred_tags, average='macro')  #, average='micro'
    f1 = f1_score(all_true_tags, all_pred_tags, average='macro')  #, average='micro'
    accuracy = accuracy_score(all_true_tags, all_pred_tags)

    # 打印结果


    return precision,recall,f1,accuracy



if __name__=="__main__":
    log = Logger(log_file_name='古汉语分词(bert_embedding)', log_level=logging.DEBUG,logger_name="debug", log_dir='./logs/', file_log=True, stream_log=True).get_log()

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--model_dir", type=str, default=r"./checkpoints/20241105")
    parser.add_argument("--trainset", type=str, default=r"./data/train.txt")
    #训练语料为全唐诗规则语料
    parser.add_argument("--validset", type=str, default=r"")
    # parser.add_argument("--testset", type=str, default=r"./data/test.txt")
    parser.add_argument("--testset", type=str, default=r"E:\南师大_计算语言学\唐诗相关\85万首诗歌语料\唐诗三百首语料\唐诗三百首分词文本\tang_300.txt")
    hp = parser.parse_args()


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Bert_BiLSTM_CRF(tag2idx).cuda()
    log.debug('Initial model Done')

    # # #单独评价
    # 检查训练模型文件是否存在
    # trained_model_path = r"H:\Pycharm Project\Local\data_process\auto_annotate_poetry\ancient_chinese_word_segment\checkpoints\20241105\best_model.pt"
    # if os.path.exists(trained_model_path):
    #     # 加载预训练模型权重进行微调
    #     model.load_state_dict(torch.load(trained_model_path))
    #     log.debug(f"Loaded pre-trained model from {trained_model_path} for fine-tuning")
    # else:
    #     log.debug(f"No pre-trained model found at {trained_model_path}, training from scratch")

    # train_dataset = NerDataset(hp.trainset)
    # eval_dataset = NerDataset(hp.validset)
    test_dataset = NerDataset(hp.testset)
    log.debug('Load Data Done')

    # train_iter = data.DataLoader(dataset=train_dataset,
    #                              batch_size=hp.batch_size,
    #                              shuffle=True,
    #                              num_workers=0,
    #                              collate_fn=pad)
    # eval_iter = data.DataLoader(dataset=eval_dataset,
    #                              batch_size=hp.batch_size,
    #                              shuffle=False,
    #                              num_workers=0,
    #                              collate_fn=pad)
    test_iter = data.DataLoader(dataset=test_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=pad)
    print(len(test_iter))
    # print(len(list(train_iter)[0][5]))




    # optimizer = optim.Adam(model.parameters(), lr = hp.lr)
    # # criterion = nn.CrossEntropyLoss(ignore_index=0)
    #
    #
    # log.debug('Start Train...,')
    # best_f1 = -1
    # best_epoch = -1
    #
    # for epoch in range(1, hp.n_epochs+1):  # 每个epoch对dev集进行测试
    #
    #     train(model, train_iter, optimizer,  device)
    #     log.debug(f"=========eval at epoch={epoch}=========")
    #     if not os.path.exists(hp.model_dir):
    #         os.makedirs(hp.model_dir)
    #     precision, recall, f1, accuracy= eval(model, test_iter, device)
    #     if f1 > best_f1:
    #         best_f1 = f1
    #         best_epoch = epoch
    #         torch.save(model.state_dict(), f"{hp.model_dir}/best_model.pt")
    #         log.debug(f"Best model weights were saved to {hp.model_dir}/best_model.pt")
    #
    #     log.debug(f"accuracy=%.4f" % accuracy)
    #     log.debug(f"precision=%.4f" % precision)
    #     log.debug(f"recall=%.4f" % recall)
    #     log.debug(f"f1=%.4f" % f1)
    #
    # log.debug(f"Best F1 score is {best_f1} at epoch {best_epoch}")




    # # #单独评价
    # precision,recall,f1,accuracy = eval(model, test_iter, device)
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1:.4f}")




