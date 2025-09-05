# -*- coding:utf-8 -*-
"""
作者：THUNDEROBOT-LI
日期：2023年12月02日
"""
import torch
from itertools import combinations
def generate_pairs(lst):
    return list(combinations(lst, 2))
def generate_train_data():
    path = '/home/y210101004/pycharm_project/langchain/data/merged_dg_list.pt'
    merge_data = torch.load(path)
    # print(len(merge_data))
    # print(merge_data)
    all_data = []
    for sub_list in merge_data:
        pairs = generate_pairs(sub_list)
        # print(pairs)
        for pair in pairs:
            all_data.append(pair[0]+'	'+pair[1]+'	'+'1')
    first_dg = []
    for sub_list in merge_data:
        if sub_list[0] not in first_dg:
            first_dg.append(sub_list[0])
    nagetive_samples = generate_pairs(first_dg)
    for pair in nagetive_samples:
        all_data.append(pair[0]+'	'+pair[1]+'	'+'0')

    all_data = list(set(all_data))
    with open('/home/y210101004/pycharm_project/langchain/text2vec-master/data/STS-B.train.data','w',encoding='utf-8') as f0:
        for i in all_data:
            f0.writelines(i+'\n')
    with open('/home/y210101004/pycharm_project/langchain/text2vec-master/data/STS-B.valid.data','w',encoding='utf-8') as f1:
        for i in all_data[-100:]:
            f1.writelines(i+'\n')
    with open('/home/y210101004/pycharm_project/langchain/text2vec-master/data/STS-B.test.data', 'w',
              encoding='utf-8') as f2:
        for i in all_data[-100:]:
            f2.writelines(i + '\n')
if __name__=="__main__":
    generate_train_data()