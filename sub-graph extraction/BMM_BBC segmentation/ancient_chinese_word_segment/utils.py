# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2019/11/07 22:11:33
@Author  :   Cao Shuai
@Version :   1.0
@Contact :   caoshuai@stu.scu.edu.cn
@License :   (C)Copyright 2018-2019, MILAB_SCU
@Desc    :   None
'''

import os
import numpy as np
import logging
import torch
from torch.utils.data import Dataset
from typing import Tuple, List
from pytorch_pretrained_bert import BertTokenizer
from transformers import AutoTokenizer, AutoModel
logger = logging.getLogger(__name__)

bert_model = r'H:\Pycharm Project\Local\data_process\auto_annotate_poetry\ancient_chinese_word_segment\bert-ancient-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_model)





VOCAB = ( '<PAD>','[CLS]', '[SEP]','B','I','E','S')  #
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}
MAX_LEN = 256 - 2


class NerDataset(Dataset):
    def __init__(self, f_path):
        with open(f_path, 'r', encoding='utf-8') as fr:
            entries = fr.read().strip().split('\n\n')
            # print(entries)
        sents, tags_li = [], [] # list of lists
        for entry in entries:
            # print(entry)
            words = [line.split()[0] for line in entry.splitlines()]  #这里报错，考虑出现了连续的大于等于2个的空行
            tags = ([line.split()[-1] for line in entry.splitlines()])
            # print(words)
            # print(tags)
            if len(words) > MAX_LEN:
                # 先对句号分段
                word, tag = [], []
                for char, t in zip(words, tags):
                    
                    if char != '。':
                        if char != '\ue236':   # 测试集中有这个字符
                            word.append(char)
                            tag.append(t)
                    else:
                        sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])
                        tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
                        word, tag = [], []            
                # 最后的末尾
                if len(word):
                    sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])
                    tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
                    word, tag = [], []
            else:
                sents.append(["[CLS]"] + words[:MAX_LEN] + ["[SEP]"])
                tags_li.append(['[CLS]'] + tags[:MAX_LEN] + ['[SEP]'])
        self.sents, self.tags_li = sents, tags_li
                

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx]
        x, y = [], []
        is_heads = []
        for w, t in zip(words, tags):



            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)
            # assert len(tokens) == len(xx), f"len(tokens)={len(tokens)}, len(xx)={len(xx)}"


            # 中文没有英文wordpiece后分成几块的情况
            is_head = [1] + [0]*(len(tokens) - 1)
            t = [t] + ['<PAD>'] * (len(tokens) - 1)
            yy = [tag2idx[each] for each in t]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)
        # print(words,len(words))
        # print(tags,len(tags))


        # if len(x)!=len(y):
        #     print(words,len(words))
        #     print(tags, len(tags))
        # assert len(x)==len(y)==len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen


    def __len__(self):
        return len(self.sents)


def pad(batch):
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(1, maxlen)
    y = f(-2, maxlen)


    f = torch.LongTensor

    return words, f(x), is_heads, tags, f(y), seqlens
    
if __name__=="__main__":
    from torch.utils.data import DataLoader, Dataset

    train_dataset = NerDataset("./data/cishu_data/train.txt")
    #VOCAB = ('<PAD>', '[CLS]', '[SEP]',  'B','I','E','O')
    for words, x, heads, tags,y,seq in DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=pad):
        print("Words:", words)
        print("Input IDs:", x)
        print("Tags:", tags)
        print("Label IDs:", y)