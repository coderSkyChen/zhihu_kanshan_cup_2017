# coding=utf-8
'''
config.py
define file path as so forth.
'''

import time

get_current_time = lambda: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\t'

DATAROOT = '/home/cxk/zhihucup/ieee_zhihu_cup_rowdata/'  #this should be written as your own data root path

# embedding files
CHAR_EMBEDDING_DIR = DATAROOT + 'char_embedding.txt'
WORD_EMBEDDING_DIR = DATAROOT + 'word_embedding.txt'

# topic info
TOPIC_INFO_DIR = DATAROOT + 'topic_info.txt'

# train and eval text
QUESTION_TRAIN_SET_DIR = DATAROOT + 'question_train_set.txt'
QUESTION_EVAL_SET_DIR = DATAROOT + 'question_eval_set.txt'

# traindata's label file
QUESTION_TOPIC_TRAIN_DIR = DATAROOT + 'question_topic_train_set.txt'
