# -*- encoding: utf-8 -*-


import torch
import os
import sys
from ancient_chinese_word_segment.utils import tag2idx, idx2tag, tokenizer
from ancient_chinese_word_segment.crf import Bert_BiLSTM_CRF


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 初始化模型
crf_model_path = r"H:\Pycharm Project\Local\data_process\auto_annotate_poetry\ancient_chinese_word_segment\checkpoints\zuozhuan_tang300_full\final_model.pt"
crf_model = Bert_BiLSTM_CRF(tag2idx)
crf_model.load_state_dict(torch.load(crf_model_path))
crf_model.to(device)
crf_model.eval()

def predict(text, model):
    """Using CRF to predict label

    Arguments:
        text {str} -- 输入文本
        model {torch.nn.Module} -- 已加载的CRF模型
    """
    tokens = ['[CLS]'] + tokenizer.tokenize(text) + ['[SEP]']
    xx = tokenizer.convert_tokens_to_ids(tokens)
    xx = torch.tensor(xx).unsqueeze(0).to(device)
    # 获取预测标签序列和标签概率分数
    _, y_hat = model(xx)
    pred_tags = [idx2tag[tag.item()] for tag in y_hat.squeeze()]
    return pred_tags, tokens

def parse(tokens, pred_tags):
    """Parse the predicted tags to real words.

    Arguments:
        tokens {List[str]} -- Original tokens.
        pred_tags {List[str]} -- Predicted tags.

    Returns:
        entities {List[str]} -- List of reconstructed words/entities.
    """
    entities = []
    idx = 0
    while idx < len(pred_tags):
        tag = pred_tags[idx]
        if tag == 'B':
            # Start a new entity from 'B'
            start = idx
            idx += 1
            while idx < len(pred_tags) and pred_tags[idx] in ['I', 'E']:
                idx += 1
            end = idx
            name = ''.join(tokens[start:end])
            entities.append(name)
        elif tag == 'E':
            # Unconditionally combine with previous character
            if idx > 0:
                name = tokens[idx - 1] + tokens[idx]
                if pred_tags[idx - 1]=='S':
                    entities.remove(entities[-1])
                entities.append(name)
                idx += 1
            else:
                # If at the first character, just append it
                entities.append(tokens[idx])
                idx += 1
        elif tag == 'S' or tag == '<PAD>'or tag == '[SEP]':
            # Single-character word
            if tokens[idx] not in ['[SEP]','<PAD>','[CLS]']:
                entities.append(tokens[idx])
            idx += 1
        else:
            # Other tags ('I', etc.), move to next character
            idx += 1
    return entities




def get_crf_cws(text, crf_model):
    pred_tags, tokens = predict(text, crf_model)
    # for p,t in zip(pred_tags, tokens):
    #     print(t,p)
    tokens = ['[CLS]'] + list(text) + ['[SEP]']
    entities = parse(tokens, pred_tags)
    return entities

if __name__ == "__main__":
    text = "芳菲全属断金人"
    print(text)
    print(get_crf_cws(text, crf_model))
