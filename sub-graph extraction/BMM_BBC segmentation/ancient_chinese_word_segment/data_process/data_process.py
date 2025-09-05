import zhconv
def process_bieo_format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in lines:
            line = line.strip()
            if line:  # 忽略空行
                # 分离词和词性标签
                words = [word.split('/')[0] for word in line.split()]

                for word in words:
                    word = zhconv.convert(word, 'zh-cn')
                    if len(word) == 1:
                        f_out.write(f"{word} S\n")  # 单字词，标记为S
                    else:
                        f_out.write(f"{word[0]} B\n")  # 第一个字，标记为B
                        for char in word[1:-1]:
                            f_out.write(f"{char} I\n")  # 中间字，标记为I
                        f_out.write(f"{word[-1]} E\n")  # 最后一个字，标记为E
                f_out.write("\n")  # 每行处理完后添加空行

def main():
    # 使用示例
    input_file = r"G:\数据集\LT4HALA-master\LT4HALA-master\2022\data_and_doc\EvaHan_testa_gold.txt"  # 输入文件路径
    output_file = r'H:\Pycharm Project\Local\bert-bilstm-crf\ancient_chinese_word_segment\data\test.txt'  # 输出文件路径
    process_bieo_format(input_file, output_file)




import random

def split_bies_data(input_file, train_file, test_file, train_ratio=0.8):
    """
    将 BIES 数据按照指定比例分成训练集和测试集。

    参数：
    - input_file: 输入的 BIES 数据文件路径。
    - train_file: 输出的训练集文件路径。
    - test_file: 输出的测试集文件路径。
    - train_ratio: 训练集所占比例，默认为 0.8（80%）。
    """
    # 读取数据并按句子分组
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sentences = []
    sentence = []
    for line in lines:
        line = line.strip()
        if line == '':
            if sentence:
                sentences.append(sentence)
                sentence = []
        else:
            sentence.append(line)
    # 如果文件末尾没有空行，确保最后一个句子被添加
    if sentence:
        sentences.append(sentence)

    # 打乱句子顺序
    random.shuffle(sentences)

    # 计算训练集和测试集的分割点
    total_sentences = len(sentences)
    train_size = int(total_sentences * train_ratio)
    train_sentences = sentences[:train_size]
    test_sentences = sentences[train_size:]

    # 将训练集写入文件
    with open(train_file, 'w', encoding='utf-8') as f:
        for sentence in train_sentences:
            for line in sentence:
                f.write(line + '\n')
            f.write('\n')  # 句子之间空行

    # 将测试集写入文件
    with open(test_file, 'w', encoding='utf-8') as f:
        for sentence in test_sentences:
            for line in sentence:
                f.write(line + '\n')
            f.write('\n')  # 句子之间空行


if __name__=="__main__":
    # main()

    # 示例用法
    input_file = 'bies_data.txt'  # 输入的 BIES 数据文件
    train_file = 'train_data.txt'  # 输出的训练集文件
    test_file = 'test_data.txt'  # 输出的测试集文件

    split_bies_data(input_file, train_file, test_file, train_ratio=0.8)