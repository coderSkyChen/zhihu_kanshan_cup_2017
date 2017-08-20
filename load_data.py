# coding=utf-8

'''
loading data
defining a class that process the loading function.
'''

import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from scipy.sparse import csr_matrix

import config


# params


class data_loader():
    def __init__(self, savedir=None):
        self.embedword_matrix = None
        self.embedword_matrix_git100 = None
        self.embedchar_matrix = None
        self.word_index = None
        self.char_index = None
        self.input_length = None
        self.topic_dict = {}
        self.topic_dict_inv = {}
        self.MAX_NB_WORDS = 500000
        self.tc_len = 180
        self.tw_len = 76
        self.max_titleword_len = 0
        self.max_titlechar_len = 0
        self.max_dspword_len = 0
        self.max_dspchar_len = 0
        self.dsppad_length = 300
        self.savedir = savedir

    def load_topic_info(self):
        '''
        just get the topic ids 
        :return: 
        '''
        print(config.get_current_time(), "loading topic info")
        with open(config.TOPIC_INFO_DIR, 'r') as f:
            for index, line in enumerate(f.readlines()):
                self.topic_dict[line.strip('\n').split('\t')[0]] = index
                self.topic_dict_inv[index] = line.strip('\n').split('\t')[0]

    def load_train_data(self):
        '''
        title_char+title_word+dsp_char+dsp_word
        :param istitle: bool 
        :param iscontent: bool
        :param type_kind: char or word
        :return: 
        '''

        title_char_list = []
        title_word_list = []
        dsp_char_list = []
        dsp_word_list = []
        question_ids = []

        print(config.get_current_time(), 'loading question train set file')
        with open(config.QUESTION_TRAIN_SET_DIR, 'r') as f:
            for index, line in enumerate(f.readlines()):
                if index > 500:
                    break
                splitted = line.strip('\n').split('\t')

                if len(splitted) == 1:
                    continue
                elif len(splitted) == 2:
                    continue
                elif len(splitted) == 5:
                    title_char_list.append(splitted[1].replace(',', ' '))
                    title_word_list.append(splitted[2].replace(',', ' '))
                    dsp_char_list.append(splitted[3].replace(',', ' '))
                    dsp_word_list.append(splitted[4].replace(',', ' '))
                    self.max_titlechar_len = max(len(splitted[1].split(',')), self.max_titlechar_len)
                    self.max_titleword_len = max(len(splitted[2].split(',')), self.max_titleword_len)
                    self.max_dspchar_len = max(len(splitted[3].split(',')), self.max_dspchar_len)
                    self.max_dspword_len = max(len(splitted[4].split(',')), self.max_dspword_len)
                    question_ids.append(splitted[0])
                else:
                    continue

        # print('max titlecharlength', self.max_titlechar_len)
        # print('max titleword length', self.max_titleword_len)
        # print('max dspchar length', self.max_dspchar_len)
        # print('max dspword length', self.max_dspword_len)

        pickle.dump(self.tw_len, open(self.savedir + '/tw_len.pkl', 'wb'))
        pickle.dump(self.tc_len, open(self.savedir + '/tc_len.pkl', 'wb'))
        pickle.dump(self.dsppad_length, open(self.savedir + '/dsp_pad_length.pkl', 'wb'))

        # ------titleword--------
        print(config.get_current_time(), 'tokenizer title word working')
        tokenizer_word = Tokenizer(num_words=self.MAX_NB_WORDS)
        tokenizer_word.fit_on_texts(title_word_list + dsp_word_list)
        sequences_titleword = tokenizer_word.texts_to_sequences(title_word_list)
        self.word_index = tokenizer_word.word_index
        print(config.get_current_time(), 'Found %s unique word tokens.' % len(self.word_index))
        titleword_array = pad_sequences(sequences_titleword, maxlen=self.tw_len)  # return arrays
        pickle.dump(tokenizer_word, open(self.savedir + '/tokenizer_word.pkl', 'wb'))
        print('tokenzier is saved as %s/tokenizer_word.pkl' % (self.savedir))
        # -----titlechar---------
        print(config.get_current_time(), 'tokenizer title char working')
        tokenizer_char = Tokenizer(num_words=self.MAX_NB_WORDS)
        tokenizer_char.fit_on_texts(title_char_list + dsp_char_list)
        sequences_titlechar = tokenizer_char.texts_to_sequences(title_char_list)
        self.char_index = tokenizer_char.word_index
        print(config.get_current_time(), 'Found %s unique char tokens.' % len(self.char_index))
        titlechar_array = pad_sequences(sequences_titlechar, maxlen=self.tc_len)  # return arrays
        pickle.dump(tokenizer_char, open(self.savedir + '/tokenizer_char.pkl', 'wb'))
        print('tokenzier is saved as %s/tokenizer_char.pkl' % (self.savedir))
        # -----dspword--------
        print(config.get_current_time(), 'tokenizer dsp char working')
        sequences_dspchar = tokenizer_char.texts_to_sequences(dsp_char_list)
        dspchar_array = pad_sequences(sequences_dspchar, maxlen=self.dsppad_length)  # return arrays
        # ---dspchar---------
        print(config.get_current_time(), 'tokenizer dsp word working')
        sequences_dspword = tokenizer_word.texts_to_sequences(dsp_word_list)
        dspword_array = pad_sequences(sequences_dspword, maxlen=self.dsppad_length)  # return arrays

        self.load_topic_info()

        question_to_label = {}
        print(config.get_current_time(), 'loading train labels')
        with open(config.QUESTION_TOPIC_TRAIN_DIR, 'r') as f:
            for index, line in enumerate(f.readlines()):
                # if index>100000:
                #     break
                splitted = line.strip('\n').split('\t')
                if len(splitted) != 2:
                    print('error!')
                question_to_label[splitted[0]] = [self.topic_dict[i] for i in splitted[1].split(',')]

        print(config.get_current_time(), 'duiqi traindata and labels')

        row_ = []
        col_ = []
        count_1 = 0
        # label_dense = np.zeros((train_titleword_array.shape[0], 1999))
        for row, quesid in enumerate(question_ids):
            cols = question_to_label.get(quesid)
            if cols is None:
                print('error!')
            count_1 += len(cols)
            for k in cols:
                row_.append(row)
            col_.extend(cols)

        data_ = [1 for i in row_]
        label_sparse = csr_matrix((data_, (row_, col_)), shape=(len(question_ids), 1999))
        # # Shuffle data
        # shuffle_indices = np.random.permutation(np.arange(train_titleword_array.shape[0]))
        # x_word = train_titleword_array[shuffle_indices]
        # x_char = train_titlechar_array[shuffle_indices]
        # row_ = [row_[i] for i in shuffle_indices]
        # col_ = [col_[i] for i in shuffle_indices]
        #
        # # label_dense = label_dense[shuffle_indices]
        # # label_sparse = csr_matrix(([1 for i in range(count_1))],(row_,col_)),shape = ())
        #
        # train_len = int(x_word.shape[0] * 0.9)
        # x_word_train = x_word[:train_len]
        # x_char_train = x_char[:train_len]
        # y_train = label_sparse[:train_len]
        # x_word_test = x_word[train_len:]
        # x_char_test = x_char[train_len:]
        # y_test = label_sparse[train_len:]

        # return (x_word_train, x_char_train, y_train, x_word_test, x_char_test, y_test)
        return titlechar_array, titleword_array, dspchar_array, dspword_array, label_sparse

    def load_pred_data_4part(self):
        '''
        
        :return: 
        '''
        title_char_list = []
        title_word_list = []
        dsp_char_list = []
        dsp_word_list = []
        question_ids = []

        self.tw_len = pickle.load(open(self.savedir + '/tw_len.pkl', 'rb'))
        self.tc_len = pickle.load(open(self.savedir + '/tc_len.pkl', 'rb'))
        self.dsppad_length = pickle.load(open(self.savedir + '/dsp_pad_length.pkl', 'rb'))
        print('length is loaded!')

        print(config.get_current_time(), 'loading question eval set file')
        with open(config.QUESTION_EVAL_SET_DIR, 'r') as f:
            for index, line in enumerate(f.readlines()):
                # if index>50000:
                #     break
                splitted = line.strip('\n').split('\t')

                if len(splitted) == 1:
                    print('error!')
                    exit()
                elif len(splitted) == 2:
                    title_char_list.append(splitted[1].replace(',', ' '))
                    title_word_list.append(" ")
                    dsp_char_list.append(" ")
                    dsp_word_list.append(" ")
                elif len(splitted) == 3:
                    title_char_list.append(splitted[1].replace(',', ' '))
                    title_word_list.append(splitted[2].replace(',', ' '))
                    dsp_char_list.append(" ")
                    dsp_word_list.append(" ")
                elif len(splitted) == 4:
                    title_char_list.append(splitted[1].replace(',', ' '))
                    title_word_list.append(splitted[2].replace(',', ' '))
                    dsp_char_list.append(splitted[3].replace(',', ' '))
                    dsp_word_list.append(" ")
                elif len(splitted) == 5:
                    title_char_list.append(splitted[1].replace(',', ' '))
                    title_word_list.append(splitted[2].replace(',', ' '))
                    dsp_char_list.append(splitted[3].replace(',', ' '))
                    dsp_word_list.append(splitted[4].replace(',', ' '))

                question_ids.append(splitted[0])

        tokenizer_word = pickle.load(open(self.savedir + '/tokenizer_word.pkl', 'rb'))
        tokenizer_char = pickle.load(open(self.savedir + '/tokenizer_char.pkl', 'rb'))
        print('tokenizer word loaded!')
        print("")

        print(config.get_current_time(), 'tokenizer working title char')
        titlechar_sequences_char = tokenizer_char.texts_to_sequences(title_char_list)
        self.char_index = tokenizer_char.word_index
        titlechar_array = pad_sequences(titlechar_sequences_char, maxlen=self.tc_len)  # return arrays

        print(config.get_current_time(), 'tokenizer working title word')
        titleword_sequences_word = tokenizer_word.texts_to_sequences(title_word_list)
        self.word_index = tokenizer_word.word_index
        titleword_array = pad_sequences(titleword_sequences_word, maxlen=self.tw_len)  # return arrays

        print(config.get_current_time(), 'tokenizer working dsp char')
        dspchar_sequences_char = tokenizer_char.texts_to_sequences(dsp_char_list)
        dspchar_array = pad_sequences(dspchar_sequences_char, maxlen=self.dsppad_length)  # return arrays

        print(config.get_current_time(), 'tokenizer working dsp word')
        dspword_sequences_word = tokenizer_word.texts_to_sequences(dsp_word_list)
        dspword_array = pad_sequences(dspword_sequences_word, maxlen=self.dsppad_length)  # return arrays

        self.load_topic_info()

        return titlechar_array, titleword_array, dspchar_array, dspword_array, question_ids

    def get_quesids(self):
        '''

        :return: 
        '''
        question_ids = []

        print(config.get_current_time(), 'loading question eval ids')
        with open(config.QUESTION_EVAL_SET_DIR, 'r') as f:
            for index, line in enumerate(f.readlines()):
                splitted = line.strip('\n').split('\t')
                question_ids.append(splitted[0])

        self.load_topic_info()
        return question_ids

    def load_wordembedding_matrix(self):

        embeddings_index = dict()

        embedding_max_value = 0
        embedding_min_value = 1

        with open(config.WORD_EMBEDDING_DIR, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                if len(line) != 257:
                    continue

                coefs = np.asarray(line[1:], dtype='float32')

                if np.max(coefs) > embedding_max_value:
                    embedding_max_value = np.max(coefs)
                if np.min(coefs) < embedding_min_value:
                    embedding_min_value = np.min(coefs)

                embeddings_index[line[0]] = coefs

        print(config.get_current_time(), ('Found %s word vectors.' % len(embeddings_index)))

        self.embedword_matrix = np.zeros((len(self.word_index) + 1, 256))
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedword_matrix[i] = embedding_vector
            else:
                self.embedword_matrix[i] = np.random.uniform(low=embedding_min_value, high=embedding_max_value,
                                                             size=256)

    def load_charembedding_matrix(self):

        embeddings_index = dict()

        embedding_max_value = 0
        embedding_min_value = 1

        with open(config.CHAR_EMBEDDING_DIR, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                if len(line) != 257:
                    continue

                coefs = np.asarray(line[1:], dtype='float32')

                if np.max(coefs) > embedding_max_value:
                    embedding_max_value = np.max(coefs)
                if np.min(coefs) < embedding_min_value:
                    embedding_min_value = np.min(coefs)

                embeddings_index[line[0]] = coefs

        print(config.get_current_time(), ('Found %s char vectors.' % len(embeddings_index)))

        self.embedchar_matrix = np.zeros((len(self.char_index) + 1, 256))
        for word, i in self.char_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedchar_matrix[i] = embedding_vector
            else:
                self.embedchar_matrix[i] = np.random.uniform(low=embedding_min_value, high=embedding_max_value,
                                                             size=256)

