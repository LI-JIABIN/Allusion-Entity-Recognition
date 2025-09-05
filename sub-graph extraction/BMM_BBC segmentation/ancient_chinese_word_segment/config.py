

import random

import torch
import os
import logging
from datetime import datetime
print(torch.cuda.is_available())
print(torch.__version__)


class Logger(object):

    def __init__(self, log_file_name, log_level, logger_name="debug", log_dir='./logs/', file_log=False,
                 stream_log=True):
        # 创建一个logger
        self.__logger = logging.getLogger(logger_name)

        # 指定日志的最低输出级别，默认为WARN级别
        self.__logger.setLevel(log_level)

        # 指定文件保存路径
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, log_file_name + '_' + str(datetime.now())[:10] + '.txt')
        if file_log == True:
            # handler1====>创建一个handler用于写入日志文件
            file_handler = logging.FileHandler(log_path)
            # 定义handler1 的输出格式
            file_formatter = logging.Formatter('[%(asctime)s] - %(levelname)s: %(message)s')
            # 把定义好的handler1的输出格式放入handler1
            file_handler.setFormatter(file_formatter)
            # 把 handler1 和 handler2 添加到 logger
            self.__logger.addHandler(file_handler)
        if stream_log == True:
            # handler2====>创建一个handler用于输出控制台
            console_handler = logging.StreamHandler()
            # 定义handler2 的输出格式
            console_formatter = logging.Formatter('-- %(levelname)s: %(message)s')
            # 把定义好的handler2的输出格式放入handler2
            console_handler.setFormatter(console_formatter)
            self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger



