# -*- coding:utf-8 -*-
"""
作者：THUNDEROBOT-LI
日期：2024年10月24日
"""
import  pandas as pd
import re
import json
from collections import OrderedDict
import os
from collections import Counter
import math
import re
import pickle
from api_calls import chatgpt_api, baidu_api

def forward_max_match(sentence, word_list):
    result = []
    i = 0
    while i < len(sentence):
        max_length = 0
        matched_word = ""
        for j in range(i + 1, min(i + 6, len(sentence) + 1)):
            word = sentence[i:j]
            if word in word_list and len(word) > max_length:
                matched_word = word
                max_length = len(word)
        if matched_word:
            result.append(matched_word)
            i += max_length
        else:
            result.append(sentence[i])  # 无匹配时仍然添加单字
            i += 1
    return result

def backward_max_match(sentence, word_list):
    result = []
    i = len(sentence)
    max_word_length = 6  # 假设最大词长为6
    while i > 0:
        matched_word = ""
        max_length = 0
        start_index = max(0, i - max_word_length)
        for j in range(i - 1, start_index - 1, -1):
            word = sentence[j:i]
            if word in word_list and len(word) > max_length:
                matched_word = word
                max_length = len(word)
                break  # 找到最长匹配后可退出循环
        if matched_word:
            result.insert(0, matched_word)
            i -= max_length
        else:
            result.insert(0, sentence[i - 1])
            i -= 1
    return result


def combined_segment(text, word_list):
    forward_result = forward_max_match(text, word_list)
    backward_result = backward_max_match(text, word_list)
    # print(forward_result)
    # print(backward_result)

    # 比较总对数概率，选择概率更大的分词结果
    if len(forward_result) < len(backward_result):
        return forward_result
    else:
        return backward_result

    #




def basic_word_dict():
    '''1.基础词典'''
    basic_dict = pd.read_csv(r"H:\Pycharm Project\Local\Crawler\搜韵网-人物\笺注资源\基础词典（汉语大词典+汉典+百度百科）\basic_dict.csv")
    basic_words = basic_dict["词汇"].tolist()
    return basic_dict,basic_words
def place_dict():
    '''2.地名'''
    place_words = []
    df_place = pd.read_excel("H:\Pycharm Project\Local\Crawler\搜韵网-人物\笺注资源\地名对照表+古今地名（辞书+网页）\place_corpus.xlsx")
    group_place_words = df_place["古地名"].apply(lambda x: x.split('、')).tolist()
    for i in group_place_words:
        place_words.extend(i)
    # print(df_place)
    # print(place_words)
    return df_place,place_words

def allusion_dict():
    '''3.典故'''
    allusions_words = []
    allusions_dict = pd.read_excel(r"H:\Pycharm Project\Local\Crawler\搜韵网-人物\笺注资源\典故（搜韵+辞书+可可诗词）\allusions.xlsx")
    group_allusions_words = allusions_dict["典故"].apply(lambda x: x.split('and')).tolist()
    for i in group_allusions_words:
        allusions_words.extend(i)
    # print(allusions_dict)
    # print(allusions_words)
    return allusions_dict,allusions_words

def position_dict():
    '''4.官职'''
    words,definitions=[], []
    with open(r"H:\Pycharm Project\Local\Crawler\搜韵网-人物\笺注资源\官职机构（辞书）\中国历代职官别名大辞典（增订本）.txt",'r',encoding='utf-8') as file:\
        lines = file.readlines()
    for line in lines:
        pairs = line.split("【释义】")
        words.append(pairs[0].strip())
        definitions.append(pairs[1].strip())
    position_dict=pd.DataFrame({"词汇":words,"释义":definitions})
    # print(position_dict)
    # print(words)
    return position_dict,words

def metonymy_dict():
    metonymy_dict = pd.read_excel("H:\Pycharm Project\Local\Crawler\搜韵网-人物\笺注资源\借代词汇（渊鉴类函+诗学含英）\借代词汇.xlsx")
    metonymy_words = []
    group_metonymy_words = metonymy_dict["借代词汇"].apply(lambda x: x.split('、')).tolist()
    for i in group_metonymy_words:
        metonymy_words.extend(i)
    # print(metonymy_dict)
    # print(metonymy_words)
    return metonymy_dict,metonymy_words


def people_dict():
    df_people = pd.read_csv("H:\Pycharm Project\Local\Crawler\搜韵网-人物\笺注资源\人物（CBDB+汉典）\people_introduction.csv")
    df_people = df_people[['人物','朝代','历代人名大辞典','唐诗大辞典_修订本','简介','全唐诗续补遗','维基','全宋诗','其他出处']]
    names = df_people["人物"].tolist()
    # print(df_people)
    # print(names)
    return df_people, names

def get_dict():
    segment_words = []
    basic_dict, basic_words = basic_word_dict()
    # print("====【1】基础词典载入完成====\n基础词典来自：1.《汉典网》爬取结果；2.汉语大词典\n")
    geographic_dict, place_words = place_dict()
    # print("====【2】地名词典载入完成====\n地名词典来自：1.《地名对照表》OCR校对；2.古今地名爬取\n")
    allusions_dict, allusions_words = allusion_dict()
    # print("====【3】典故词典载入完成====\n典故词典来自：1.搜韵网典故爬取；2.辞书《全唐诗典故辞典》；3.可可诗词网爬取\n")
    positions_dict, position_words = position_dict()
    # print("====【4】官职名词典载入完成====\n地名词典来自：1.辞书中国历代职官别名大辞典（增订本）\n")
    metonymys_dict, metonymy_words = metonymy_dict()
    # print("====【5】借代词典载入完成====\n地名词典来自：1.诗学含英；2.渊鉴类函\n")
    character_dict, character_names = people_dict()
    # print("====【6】人名词典载入完成====\n地名词典来自：1.CBDB历代人物传记；2.历代人名大辞典\n")
    segment_words.extend(basic_words)
    segment_words.extend(place_words)
    segment_words.extend(allusions_words)
    segment_words.extend(position_words)
    segment_words.extend(metonymy_words)
    segment_words.extend(character_names)
    segment_words =  [word for word in segment_words if isinstance(word, str)]
    return basic_dict,basic_words,geographic_dict,place_words,allusions_dict,allusions_words,\
           positions_dict,position_words,metonymys_dict,metonymy_words,\
           character_dict,character_names,segment_words



#找到所有
# def find_source_and_definition(word, basic_dict, basic_words, geographic_dict, place_words, allusions_dict, allusions_words, \
#     positions_dict, position_words, metonymys_dict, metonymy_words, \
#     character_dict, character_names):
#     '''查找词汇来源及释义，收集所有匹配结果'''
#     results = []  # 存储所有匹配结果
#
#     if word in allusions_words:
#         matched_row = allusions_dict[allusions_dict["典故"].str.contains(word)]
#         js = re.sub("\n", "", matched_row["简释"].values[0])
#         cc = re.sub("\n", "", matched_row["出处"].values[0])
#
#         # 检查是否为 NaN 或浮点数，若是则设置为空字符串
#         example_sentence = matched_row["例句"].values[0] if not pd.isna(matched_row["例句"].values[0]) else ""
#         related_person = matched_row["相关人物"].values[0] if not pd.isna(matched_row["相关人物"].values[0]) else ""
#
#         # 生成 label，使用切片操作确保字符串不超过100字符
#         label = f'出处：{cc}\n简释：{js}\n例句：{example_sentence[:100]}\n相关人物：{related_person[:100]}\n'
#         results.append({"来源": "典故词典", "释义": label})
#
#     if word in basic_words:
#         definition = basic_dict[basic_dict["词汇"] == word]["释义"].values[0]
#         results.append({"来源": "基础词典", "释义": definition})
#
#     if word in place_words:
#         matched_row = geographic_dict[geographic_dict["古地名"].str.contains(word)]
#         label = f'现地名：{matched_row["现地名"].values[0]}；历史沿革信息：{matched_row["历史沿革信息"].values[0]}；'
#         results.append({"来源": "地名词典", "释义": label})
#
#     if word in position_words:
#         definition = positions_dict[positions_dict["词汇"] == word]["释义"].values[0]
#         results.append({"来源": "官职词典", "释义": definition})
#
#     if word in metonymy_words:
#         matched_row = metonymys_dict[metonymys_dict["借代词汇"].str.contains(word)]
#         label = f'一级标签：{matched_row["一级标签"].values[0]}；二级标签：{matched_row["二级标签"].values[0]}'
#         results.append({"来源": "借代词典", "释义": label})
#
#     if word in character_names:
#         definition = character_dict[character_dict["人物"] == word]["历代人名大辞典"].values[0]
#         results.append({"来源": "人物词典", "释义": definition})
#
#     # 如果没有任何匹配结果，则返回默认的“未知”结果
#     if not results:
#         results.append({"来源": "未知", "释义": "无释义"})
#
#     return results




# def segment_poem(poem, basic_dict, basic_words, geographic_dict, place_words, allusions_dict, allusions_words, \
#                  positions_dict, position_words, metonymys_dict, metonymy_words, \
#                  character_dict, character_names, segment_words):
#     seg_sen = combined_segment(poem, segment_words)
#     segmented_with_definitions = []
#     for word in seg_sen:
#         if word not in '，。？！,.?!《》' and len(word) >= 2:
#             matches = find_source_and_definition(word, basic_dict, basic_words, geographic_dict, place_words,
#                                                  allusions_dict, allusions_words, \
#                                                  positions_dict, position_words, metonymys_dict, metonymy_words, \
#                                                  character_dict, character_names)
#
#             # 处理所有匹配到的结果
#             for match in matches:
#                 segmented_with_definitions.append({
#                     "词汇": word,
#                     "来源": match['来源'],
#                     "释义": match['释义']
#                 })
#
#     return segmented_with_definitions



def ZhengDingZhuShi(poem):
    def find_recent_period_indices(text):
        indices = []
        prev_period_index = -1

        # 遍历文本中的每个":"的索引
        for i, char in enumerate(text):
            if char == "：":
                # 查找当前":"前面出现的最近的"."的索引
                period_index = text.rfind("。", 0, i)
                indices.append(period_index)

        return indices
    def generate_ngrams(text):
        ngrams = []
        for i in range(1, len(text)):
            start = 0
            end = start + i
            while end <= len(text):
                span = text[start:end]  # str_span_6: '三十三人椀杖全'
                ngrams.append(span)
                start += 1
                end += 1
            start, end = 0, 0
        return ngrams
    poem = re.sub('\s+','',poem)
    sen_peom = re.split(r'[，。,.?!！？：]+',poem)  #多多句诗进行分割
    # print(sen_peom)
    sen_peom = [i for i in sen_peom if i]
    poem_spans = []
    for sen in sen_peom:#生成所有诗句的ngrams
        spans = generate_ngrams(sen)
        poem_spans.extend(spans)
    # print(poem_spans)
    with open(r"H:\Pycharm Project\Local\Crawler\搜韵网-人物\笺注资源\翻译+赏析+专家笺注\增订注释全唐诗整理后.txt",'r',encoding='utf-8') as f0:
        paras = f0.read().split('\n\n')
    xuhao = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕㉖㉘㉙㉚㉛"
    words_anno = []
    for para in paras:
        if para:
            if max(sen_peom, key=len) in para:  #以诗句中长度最长的那句为标准在整个增订注释全唐诗中进行匹配
                poem_lines = para.split('\n')
                poem_lines = [i for i in poem_lines if i]
                poem_anno = [i for i in poem_lines if i[0]=='①']
                zhushi = re.sub('|'.join(list(xuhao)),'',''.join(poem_anno))
                # print(zhushi)
                juhao_idx = find_recent_period_indices(zhushi)  # 找出每一个冒号前面的句号的索引
                juhao_idx = [i for i in juhao_idx if int(i) > 0]
                if len(juhao_idx)>=1:
                    # print(juhao_idx)
                    start = 0
                    end = 0
                    end += 1
                    words_anno.append(zhushi[:juhao_idx[0]])
                    # print(words_anno)
                    while end < len(juhao_idx):
                        s = juhao_idx[start] + 1
                        e = juhao_idx[end] + 1
                        words_anno.append(zhushi[s:e])
                        start += 1
                        end += 1
                    words_anno.append(zhushi[juhao_idx[-1] + 1:])
    # print(words_anno)
    words_anno = [i for i in words_anno if i]
    final_words_anno = []
    for span in poem_spans:
        for anno in words_anno:
            if span in re.split('：',anno)[0]:
                # print(span)
                final_words_anno.append(anno)
    final_words_anno = list(OrderedDict.fromkeys(final_words_anno))  #对列表进行去重后还保留原来的顺序
    return final_words_anno

def gushiwen(text):
    def find_recent_period_indices(text):
        indices = []
        prev_period_index = -1

        # 遍历文本中的每个":"的索引
        for i, char in enumerate(text):
            if char == "：":
                # 查找当前":"前面出现的最近的"."的索引
                period_index = text.rfind("。", 0, i)
                indices.append(period_index)

        return indices
    path = "H:\Pycharm Project\Local\Crawler\搜韵网-人物\笺注资源\翻译+赏析+专家笺注\古诗文网.xlsx"
    pd_gushiwen = pd.read_excel(path)
    poem = re.sub('\s+', '', text)
    sen_peom = re.split('，|\.|,|。|\?|？|！|!|：|:', poem)  # 多多句诗进行分割
    sen_peom = [i for i in sen_peom if i]
    author, some_zhushi, some_fanyi, zuozhexinxi, shangxi, chuangzuobeijing = "",[],"","","",""
    for idx,poem in enumerate(pd_gushiwen["内容"].to_list()):
        if sen_peom[-1] in str(poem):
            row = pd_gushiwen.iloc[idx,:]
            title = row[0]
            author = row[1]
            content = re.sub('\(.*?\)','',row[2])
            zhushi = row[3]
            fanyi = row[4]
            zuozhexinxi = row[5]
            shangxi = row[6]
            chuangzuobeijing = row[7]
            content_lst = re.split('，|\.|,|。|\?|？|！|!', content)
            content_lst = [i for i in content_lst if i]

            some_zhushi = []
            some_fanyi = []
            if isinstance(zhushi,str):
                zhushi = re.sub('\n','',zhushi)
                juhao_idx = find_recent_period_indices(zhushi)  #找出每一个冒号前面的句号的索引
                juhao_idx = [i for i in juhao_idx if int(i)>0]
                start = 0
                end = 0
                end += 1
                # print(zhushi)
                if len(juhao_idx)>1:
                    some_zhushi.append(zhushi[ :juhao_idx[0]])
                    while end < len(juhao_idx):
                        s = juhao_idx[start]+1
                        e = juhao_idx[end]+1
                        # print(s,e)
                        some_zhushi.append(zhushi[s:e])
                        # print(some_zhushi)
                        start+=1
                        end+=1
                    some_zhushi.append(zhushi[juhao_idx[-1]+1:])
                else:
                    some_zhushi.append(zhushi)
                fanyi_lst = fanyi.split('\n')
                # print(fanyi_lst)
                for sen_txt in sen_peom:
                    max_overlap = 0
                    best_translation = None
                    for sen_fanyi in fanyi_lst:
                            overlap = len(set(sen_txt) & set(sen_fanyi))
                            if overlap > max_overlap:
                                max_overlap = overlap
                                best_translation = sen_fanyi
                    if best_translation not in some_fanyi:
                        if best_translation:
                            some_fanyi.append(best_translation)
                sen_fanyi = ''.join(sen_fanyi)
            continue
    # print(some_zhushi)
    annotation = f"{'；'.join(some_zhushi)}"\
                # f"翻译：{''.join(some_fanyi)}\n"\
                #  f"作者：{author}\n" \
                #  f"作者信息：{zuozhexinxi}\n" \
                #  f"赏析：{shangxi}\n" \
                #  f"创作背景：{chuangzuobeijing}\n"
    return annotation

# if __name__=="__main__":
#
#     # 古诗文网
#     from prompt_setting import gushiwen_prompt, zengding_prompt, lr_prompt, fianl_prompt
#
#     #    1.当存在来自《增订注释全唐诗》或《古诗文网》的笺注结果时，优先从《增订注释全唐诗》、《古诗文网》中进行提取。如果是典故则无条件提取语言资源的笺注结果。
#
#     basic_dict, basic_words, geographic_dict, place_words, allusions_dict, allusions_words, \
#     positions_dict, position_words, metonymys_dict, metonymy_words, \
#     character_dict, character_names, segment_words = get_dict()
#     all_annotations = ""
#     poem = "安乐公主移入新宅侍宴应制:星桥他日创,仙榜此时开。马向铺钱埒，箫闻弄玉台。人同卫叔美,客似长卿才。借问游天汉，谁能取石回?"
#     # "安乐公主移入新宅侍宴应制:星桥他日创,仙榜此时开。马向铺钱埒，箫闻弄玉台。人同卫叔美,客似长卿才。借问游天汉，谁能取石回?"  #和曲典设扈从东郊忆弟使往安西冬至日恨不得同申拜庆
#     #"甘露殿侍宴应制：月宇临丹地，云窗网碧纱。御筵陈桂醑，天酒酌榴花。水向浮桥直，城连禁苑斜。承恩恣欢赏，归路满烟霞。"
#     # "和麹典设扈从东郊忆弟使往安西冬至日恨不得同申拜庆玉关方叱驭，桂苑正陪舆。桓岭嗟分翼，姜川限馈鱼。雪花含□晚，云叶带荆舒。重此西流咏，弥伤南至初。"
#
#     #增订注释全唐诗笺注整合
#     final_words_anno = ZhengDingZhuShi(poem)
#     ZhengDingZhuShi = '；'.join(final_words_anno)
#     zd_annotations = ""
#     # if len(ZhengDingZhuShi)>4:
#     #     zd_query = zengding_prompt + '\n' + '内容为：' + poem + '已有信息：\n' + ZhengDingZhuShi
#     #     zd_annotations = baidu_api(zd_query)
#     #     # zd_annotations = chatgpt_api(zd_query)
#     #     zd_annotations ='《增订注释全唐诗》笺注结果：'+ '\n'  + zd_annotations+'\n'
#     #     # print(zd_annotations)
#
#     # 古诗文网笺注整合
#     # 基座 预训练  RAG
#     # gushiwenwang = gushiwen(poem)
#     # gsw_annotations = ""
#     # if len(gushiwenwang) > 4:
#     #     gsw_query = gushiwen_prompt + '\n' + '内容为：\n' + poem + '已有信息：\n' + gushiwenwang
#     #     gsw_annotations = baidu_api(gsw_query)
#     #     # gsw_annotations = chatgpt_api(gsw_query)
#     #     gsw_annotations = '《古诗文网》笺注结果：'+ '\n'  + gsw_annotations + '\n'
#     #     # print(gsw_annotations)
#
#     # 语言资源笺注
#     seg_sen = segment_poem(poem, basic_dict, basic_words, geographic_dict, place_words, allusions_dict, allusions_words, \
#     positions_dict, position_words, metonymys_dict, metonymy_words, \
#     character_dict, character_names, segment_words)
#
#     # # 语言资源笺注整合
#     lr_query = lr_prompt + '\n' + '内容为：' + poem + '已有信息：\n' + seg_sen
#     lr_annotations = baidu_api(lr_query)
#     # lr_annotations = chatgpt_api(lr_query)
#     lr_annotations = '语言资源笺注结果：'+ '\n'  + lr_annotations + '\n'
#
#     #最终
#     final_annos = zd_annotations + gsw_annotations + lr_annotations + '\n'
#     final_query = fianl_prompt + '\n' + '内容为：' + poem + '已有信息：\n' + final_annos
#     final_annotations = baidu_api(final_query)
#     # final_annotations =chatgpt_api(final_query)
#     final_annotations = '最终笺注结果：'+ '\n'  +final_annotations + '\n'
#     print(final_annos)
#     print(final_annotations)



