# -*- coding:utf-8 -*-
"""
作者：THUNDEROBOT-LI
日期：2024年11月04日
"""
import re
from ancient_chinese_word_segment.crf_predict import get_crf_cws
from dict_load import get_dict, combined_segment
import numpy as np
import torch
import os
import pandas  as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

basic_dict, basic_words, geographic_dict, place_words, allusions_dict, allusions_words, \
positions_dict, position_words, metonymys_dict, metonymy_words, \
character_dict, character_names, segment_words = get_dict()
from ancient_chinese_word_segment.crf_predict import crf_model,tokenizer,tag2idx,parse,idx2tag
from tqdm import tqdm


def bmm_segment(text, segment_words):
    seg_sen = combined_segment(text, segment_words)
    return seg_sen


def bert_bilstm_crf_segment(text):
    seg_sen, crf_probabilities = get_crf_cws(text)
    return seg_sen, crf_probabilities

def bmm_segmentation_to_labels(text, dict_segmentation, label_to_index):
    #将BMM分词结果转换为标签序列
    label_dict = {}
    pos = 0
    for word in dict_segmentation:
        word_len = len(word)
        if word_len == 1:
            label_dict[pos] = label_to_index['S']
        else:
            label_dict[pos] = label_to_index['B']
            for i in range(1, word_len - 1):
                label_dict[pos + i] = label_to_index['I']
            label_dict[pos + word_len - 1] = label_to_index['E']
        pos += word_len
    # 将标签字典转换为列表
    labels = []
    for i in range(len(text)):
        if i in label_dict:
            labels.append(label_dict[i])
        else:
            labels.append(label_to_index['O'])  # 如果不存在，标记为'O'
    return labels



def fusion_predict(text, model, dict_segmentation, alpha=1.0):
    """使用融合策略进行预测

    Arguments:
        text {str} -- 输入文本
        model {torch.nn.Module} -- 已加载的 CRF 模型
        dict_segmentation {List[str]} -- BMM 分词结果
        alpha {float} -- 调整因子
    """

    tokens = ['[CLS]'] + tokenizer.tokenize(text) + ['[SEP]']
    xx = tokenizer.convert_tokens_to_ids(tokens)
    xx = torch.tensor(xx).unsqueeze(0).to(device)
    # 获取 LSTM 特征
    lstm_feats = model._get_lstm_features(xx)
    # 将 BMM 分词结果转换为标签序列
    bmm_labels = bmm_segmentation_to_labels(text, dict_segmentation, tag2idx)
    bmm_labels = [tag2idx['[CLS]']] + bmm_labels + [tag2idx['[SEP]']]
    bmm_labels = torch.tensor(bmm_labels).unsqueeze(0).to(device)
    # print(bmm_labels)
    # 获取模型的初始预测
    _, y_hat = model._viterbi_decode(lstm_feats)
    y_hat = y_hat.to(device)
    # 将预测标签和 BMM 标签从索引转换为实际标签名
    # pred_tags = [idx2tag[tag.item()] for tag in y_hat.squeeze()]
    # bmm_tags = [idx2tag[tag.item()] for tag in bmm_labels.squeeze()]
    # 初始化需要调整的位置列表
    adjust_positions_alpha = []
    adjust_positions_100 = []
    # 初始化需要调整的位置列表
    adjust_positions_50 = []
    pos = 0
    for word in dict_segmentation:
        word_len = len(word)
        bmm_labels_word = bmm_labels[0, pos+1:pos+1+word_len]  # +1 是因为有 [CLS]
        # print(bmm_labels_word)
        y_hat_labels_word = y_hat[0, pos+1:pos+1+word_len]
        # print(y_hat_labels_word)
        # 如果词长度大于等于 4，且模型预测与 BMM 标签不一致，则需要调整 100
        if word_len >= 3:
            if not torch.equal(bmm_labels_word, y_hat_labels_word):
                # 记录需要调整的位置（增加 100）
                for i in range(word_len):
                    adjust_positions_100.append(pos + i + 1)  # +1 是因为有 [CLS]
        # 如果词长度为 2 或 3，且模型预测与 BMM 标签不一致，则需要调整 alpha
        # if  1<word_len<3:
        #     if not torch.equal(bmm_labels_word, y_hat_labels_word):
        #         # 记录需要调整的位置（增加 alpha）
        #         for i in range(word_len):
        #             adjust_positions_alpha.append(pos + i + 1)  # +1 是因为有 [CLS]
        # # 如果词长度大于 1（多字词）
        if 1<word_len<3:
            # 将标签索引转换为实际标签
            bmm_tags = [idx2tag[tag.item()] for tag in bmm_labels_word]
            y_hat_tags = [idx2tag[tag.item()] for tag in y_hat_labels_word]
            # 检查 BMM 标签是否为一个完整的词（非全 'S'）
            if not all(tag == 'S' for tag in bmm_tags):
                # 检查模型预测是否将该词拆分为多个单字词（全为 'S'）
                if all(tag == 'S' for tag in y_hat_tags):
                    # 记录需要调整的位置（增加 50）
                    for i in range(word_len):
                        adjust_positions_50.append(pos + i + 1)  # +1 是因为有 [CLS]
        pos += word_len
    # 调整发射得分
    # print(lstm_feats)
    for idx in adjust_positions_100:
        lstm_feats[0, idx, bmm_labels[0, idx]] += 100
    # for idx in adjust_positions_alpha:
    #     lstm_feats[0, idx, bmm_labels[0, idx]] += alpha
    # 调整发射得分
    for idx in adjust_positions_50:
        lstm_feats[0, idx, bmm_labels[0, idx]] += 20
    # 重新进行维特比解码
    score, y_hat = model._viterbi_decode(lstm_feats)
    # 计算边缘概率
    tag_probabilities = model._compute_marginal_probabilities(lstm_feats)
    pred_tags = [idx2tag[tag.item()] for tag in y_hat.squeeze()]
    return pred_tags, tokens, tag_probabilities

def fusion_bmm_bertbilstmcrf(text):
    # 获取BMM分词结果
    dict_segmentation = bmm_segment(text, segment_words)
    # print('BMM分词：','/'.join(dict_segmentation))
    bmm_seg = '/'.join(dict_segmentation)
    bertbilstmcrf_entities = get_crf_cws(text, crf_model)
    # print('bertbilstmcrf分词：','/'.join(bertbilstmcrf_entities))
    bbc_seg = '/'.join(bertbilstmcrf_entities)

    pred_tags, tokens, tag_probabilities = fusion_predict(text, crf_model, dict_segmentation, alpha=20)
    tokens = ['[CLS]']+list(text)+['[SEP]']  #把tokens变成text中的字，不然会出现[UNK]
    # for p,t in zip(pred_tags, tokens):
    #     print(t,p)
    entities = parse(tokens, pred_tags)
    # print('Fusion分词：', '/'.join(entities))
    # print(tag_probabilities.shape)
    fusion_seg = '/'.join(entities)
    return bmm_seg,bbc_seg,fusion_seg
if __name__ == "__main__":
    # text = "原陵老翁吟 其二 狐书一：正色鸿焘，神思化伐。穹施后承，光负玄设。呕沦吐萌，垠倪散截。迷肠郗曲，䨴零霾曀。雀燬龟水，健驰御屈。拿尾研动，袾袾??。㳷用秘功，以岭以穴。柂薪伐药，莽桀万茁。呕律则祥，佛伦惟萨。牡虚无有，颐咽蕊屑。肇素未来，晦明兴灭。"
    text = "芳菲全属断金人"
    bmm_seg,bbc_seg,fusion_seg = fusion_bmm_bertbilstmcrf(text)
    print(fusion_seg)





'''
"于北平作:翠野驻戎轩，卢龙转征旃。遥山丽如绮，长流萦似带。海气百重楼，岩松千丈盖。兹馬可游赏，何必襄城外。"
"安乐公主移入新宅侍宴应制:星桥他日创,仙榜此时开。马向铺钱埒，箫闻弄玉台。人同卫叔美,客似长卿才。借问游天汉，谁能取石回?"
"奉和圣制喜雪应制:飘飘瑞雪下山川，散漫轻飞集九蜒。似絮还飞垂柳陌,如花更绕落梅前。影随明月团纨扇,声将流水杂鸣弦。共荷神功万庾积,终朝圣寿百千年。"
"送崔主簿赴沧州:紫陌追随日，青门相见时。宦游从此去，离别几年期。芳桂尊中酒，幽兰下调词。他乡有明月，千里照相思。"
"十一月奉教作:凝阴结暮序,严气肃长飙。霜犯狐裘夕，寒侵兽火朝。冰深逢架浦，雪冻近封条。平原已从猎，日暮整还镳。"
"二月奉教作:柳陌莺初啭，梅梁燕始归。和风泛紫若,柔露濯青薇。日艳临花影，霞翻入浪晖。乘春重游豫,淹赏玩芳菲。"
"十二月奉教作:玉烛年行尽,铜史漏犹长。池冷凝宵冻，庭寒积曙霜。兰心未动色，梅馆欲含芳。裴回临岁晚，顾步伫春光。"
"武三思挽歌:玉匣金为缕,银钩石作铭。短歌伤薤曲,长暮泣松扃。事往昏朝雾，人亡折夜星。忠贤良可惜，图画入丹青。"
"天官崔侍郎夫人吴氏挽歌:宠服当年盛,芳魂此地穷。剑飞龙匣在,人去鹊巢空。簟怆孤生竹,琴哀半死桐。唯当青史上，千载仰嫔风。"
"和麹典设扈从东郊忆弟使往安西冬至日恨不得同申拜庆:玉关方叱驭，桂苑正陪舆。桓岭嗟分翼，姜川限馈鱼。雪花含□晚，云叶带荆舒。重此西流咏，弥伤南至初。"
'''