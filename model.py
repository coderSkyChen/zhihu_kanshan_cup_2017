# encoding: utf-8

import math
import sys

import config
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.layers import Embedding, merge, Reshape, Activation, RepeatVector, Permute, Lambda, GlobalMaxPool1D, \
    concatenate
from keras import initializers
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.layers import Dense, Conv1D, MaxPooling1D, Input, Flatten, Dropout, Concatenate, LSTM, Bidirectional, GRU
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.models import load_model
from keras.layers.normalization import BatchNormalization

from load_data import data_loader


class ZHIHUMetrics(Callback):
    '''
    ZHIHU score method
    '''

    def on_epoch_end(self, batch, logs={}):
        print('')
        y_pred = np.asarray(self.model.predict(
            [self.validation_data[0], self.validation_data[1], self.validation_data[2], self.validation_data[3]]))
        y_true = self.validation_data[4]
        # y_pred = np.asarray(self.model.predict([self.validation_data[0], self.validation_data[1]]))
        # y_true = self.validation_data[2]
        # y_pred = np.asarray(self.model.predict([self.validation_data[0]]))
        # y_true = self.validation_data[1]

        print(y_pred.shape, y_true.shape)

        y_pred = np.argsort(-y_pred)[:, :5]

        y_true_list = []
        for i in range(y_pred.shape[0]):
            y_true_list.append([])

        nozero_row, nozero_col = np.nonzero(y_true)

        for i in range(len(nozero_row)):
            y_true_list[nozero_row[i]].append(nozero_col[i])

        right_label_num = 0
        right_label_at_pos_num = [0, 0, 0, 0, 0]
        sample_num = 0
        all_marked_label_num = 0

        for i in range(len(y_true_list)):
            sample_num += 1
            marked_label_set = set(y_true_list[i])
            all_marked_label_num += len(marked_label_set)
            for pos, label in zip(range(0, min(len(y_pred[i]), 5)), y_pred[i]):
                if label in marked_label_set:
                    right_label_num += 1
                    right_label_at_pos_num[pos] += 1

        precision = 0.0
        for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
            precision += ((right_num / float(sample_num))) / math.log(2.0 + pos)
        recall = float(right_label_num) / all_marked_label_num

        print('Recall:', recall)
        print(' Precision:', precision)
        print(' res:', recall * precision / (recall + precision + 0.00000000000001))
        print('')


class MultiModel():
    def __init__(self, w_embed_matrix=None, c_embed_matrix=None, word_index=None, char_index=None,
                 titlechar_length=None, titleword_length=None, dsp_padlen=None, data=(0, 0, 0), savedir=None):
        # Model Hyperparameters
        self.hidden_dims = 512
        self.EMBEDDING_DIM = 256
        # Training parameters
        self.batch_size = 128
        self.num_epochs = 50

        self.w_embed = w_embed_matrix
        self.c_embed = c_embed_matrix
        self.word_index = word_index
        self.char_index = char_index
        self.titlechar_length = titlechar_length
        self.titleword_length = titleword_length
        self.dsp_padlen = dsp_padlen

        # data
        if len(data) == 5:
            self.titlechar_array, self.titleword_array, self.dspchar_array, self.dspword_array, self.y = data

        self.savedir = savedir

        self.model = None

    def buildmodel_rcnn4_att_titledsp(self):
        '''
        4 RCNN
        v2: 4model concat+dense1999
        (tw concat tc)  +  (dw concat dc)
        lstm256+lr0.001  :3epoch 0.401  0.9data
        lstm512+lr0.0005  :2epoch 0.410  alldata    2,3,4 epoch vote  0.414   with dp

        :return: 
        '''
        print('building model...')

        # -----titlechar------
        with tf.device('/cpu:%d' % (0)):
            tc_embedding_layer = Embedding(len(self.char_index) + 1,
                                           self.EMBEDDING_DIM,
                                           weights=[self.c_embed],
                                           input_length=self.titlechar_length, trainable=True,
                                           embeddings_initializer=initializers.RandomUniform(minval=-0.2, maxval=0.2,
                                                                                             seed=None))
        tc_sequence_input = Input(shape=(self.titlechar_length,), name="titlechar_input")
        tc_embedded_sequences = tc_embedding_layer(tc_sequence_input)
        with tf.device('/gpu:%d' % (0)):
            tc_z_pos = LSTM(512, implementation=2, return_sequences=True, go_backwards=False)(tc_embedded_sequences)
            tc_z_neg = LSTM(512, implementation=2, return_sequences=True, go_backwards=True)(tc_embedded_sequences)
            tc_z_concat = merge([tc_z_pos, tc_embedded_sequences, tc_z_neg], mode='concat', concat_axis=-1)

            tc_z = Dense(512, activation='tanh')(tc_z_concat)
            tc_pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(512,))(tc_z)
        # -----titleword------
        with tf.device('/cpu:%d' % (1)):
            tw_embedding_layer = Embedding(len(self.word_index) + 1,
                                           self.EMBEDDING_DIM,
                                           weights=[self.w_embed],
                                           input_length=self.titleword_length, trainable=True,
                                           embeddings_initializer=initializers.RandomUniform(minval=-0.2, maxval=0.2,
                                                                                             seed=None))
        tw_sequence_input = Input(shape=(self.titleword_length,), name="titleword_input")
        tw_embedded_sequences = tw_embedding_layer(tw_sequence_input)
        with tf.device('/gpu:%d' % (0)):
            tw_z_pos = LSTM(512, implementation=2, return_sequences=True, go_backwards=False)(tw_embedded_sequences)
            tw_z_neg = LSTM(512, implementation=2, return_sequences=True, go_backwards=True)(tw_embedded_sequences)
            tw_z_concat = merge([tw_z_pos, tw_embedded_sequences, tw_z_neg], mode='concat', concat_axis=-1)

            tw_z = Dense(512, activation='tanh')(tw_z_concat)
            tw_pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(512,))(tw_z)
        # -----dspchar------
        with tf.device('/cpu:%d' % (2)):
            dc_embedding_layer = Embedding(len(self.char_index) + 1,
                                           self.EMBEDDING_DIM,
                                           weights=[self.c_embed],
                                           input_length=self.dsp_padlen, trainable=True,
                                           embeddings_initializer=initializers.RandomUniform(minval=-0.2, maxval=0.2,
                                                                                             seed=None))
        dc_sequence_input = Input(shape=(self.dsp_padlen,), name="dspchar_input")
        dc_embedded_sequences = dc_embedding_layer(dc_sequence_input)
        with tf.device('/gpu:%d' % (1)):
            dc_z_pos = LSTM(512, implementation=2, return_sequences=True, go_backwards=False)(dc_embedded_sequences)
            dc_z_neg = LSTM(512, implementation=2, return_sequences=True, go_backwards=True)(dc_embedded_sequences)
            dc_z_concat = merge([dc_z_pos, dc_embedded_sequences, dc_z_neg], mode='concat', concat_axis=-1)
            dc_z = Dense(512, activation='tanh')(dc_z_concat)
            dc_pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(512,))(dc_z)
        # -----dspword------
        with tf.device('/cpu:%d' % (3)):
            dw_embedding_layer = Embedding(len(self.word_index) + 1,
                                           self.EMBEDDING_DIM,
                                           weights=[self.w_embed],
                                           input_length=self.dsp_padlen, trainable=True,
                                           embeddings_initializer=initializers.RandomUniform(minval=-0.2, maxval=0.2,
                                                                                             seed=None))
        dw_sequence_input = Input(shape=(self.dsp_padlen,), name="dspword_input")
        dw_embedded_sequences = dw_embedding_layer(dw_sequence_input)
        with tf.device('/gpu:%d' % (1)):
            dw_z_pos = LSTM(512, implementation=2, return_sequences=True, go_backwards=False)(dw_embedded_sequences)
            dw_z_neg = LSTM(512, implementation=2, return_sequences=True, go_backwards=True)(dw_embedded_sequences)
            dw_z_concat = merge([dw_z_pos, dw_embedded_sequences, dw_z_neg], mode='concat', concat_axis=-1)

            dw_z = Dense(512, activation='tanh')(dw_z_concat)
            dw_pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(512,))(dw_z)

        # ------att----------
        concat_w_c = merge([tc_pool_rnn, tw_pool_rnn, dc_pool_rnn, dw_pool_rnn], mode='concat')
        concat_w_c = Reshape((2, 512 * 2))(concat_w_c)

        attention = Dense(1, activation='tanh')(concat_w_c)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(512 * 2)(attention)
        attention = Permute([2, 1])(attention)

        sent_representation = merge([concat_w_c, attention], mode='mul')
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(512 * 2,))(sent_representation)
        # --------merge_4models------------------
        model_final_ = Dense(1999, activation='relu')(sent_representation)
        model_final_ = Dropout(0.5)(model_final_)
        model_final = Dense(1999, activation='softmax')(model_final_)

        self.model = Model(input=[tc_sequence_input, tw_sequence_input, dc_sequence_input, dw_sequence_input],
                           outputs=model_final)
        adam = optimizers.adam(lr=0.0001)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=adam,
                           metrics=[categorical_accuracy])
        print(self.model.summary())

    def trainmodel(self, isalldata):

        self.buildmodel_rcnn4_att_titledsp()

        import time
        cur_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))

        checkpointer = ModelCheckpoint(filepath=self.savedir + "/" + cur_time + "_model-{epoch:02d}.hdf5", period=1)
        zhihuMetrics = ZHIHUMetrics()

        if isalldata:
            self.model.fit([self.titlechar_array, self.titleword_array, self.dspchar_array, self.dspword_array],
                           self.y,
                           epochs=self.num_epochs, batch_size=self.batch_size, verbose=1,
                           callbacks=[checkpointer])
        else:#with 9:1 validation
            self.model.fit([self.titlechar_array, self.titleword_array, self.dspchar_array, self.dspword_array],
                           self.y,
                           validation_split=0.1,
                           epochs=self.num_epochs, batch_size=self.batch_size, verbose=1,
                           callbacks=[checkpointer, zhihuMetrics])
        self.save_model()

    def predmodel(self, modelname, datatuple, topic_dict_inv):

        import time
        cur_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
        from collections import Counter

        def tmpfunc(x):
            if len(x) > 5:
                c = Counter(x).most_common(5)
                res = []
                for num, count in c:
                    res.append(topic_dict_inv[num])
            else:
                res = []
                for i in x:
                    res.append(topic_dict_inv[i])

            return res

        predlabels = []
        # titleword_array, dspword_array, ques_ids= datatuple
        titlechar_array, titleword_array, dspchar_array, dspword_array, ques_ids = datatuple

        for i in range(len(modelname)):
            self.model = load_model(modelname[i])
            predlabel = self.model.predict([titlechar_array, titleword_array, dspchar_array, dspword_array],
                                           batch_size=512, verbose=1)
            # predlabel = self.model.predict([titleword_array, titleword_array, dspword_array, dspword_array], batch_size=512, verbose=1)
            # np.savetxt("result/scores/"+cur_time + "scores_4RCNN_gru_dense_nodropout.txt", predlabel, fmt='%s')
            np.save("result/scores/" + cur_time + "4RCNN_lstm512_4part_title_dsp_attention_nofc_06epoch", predlabel)
            # exit()
            predlabel = np.argsort(-predlabel)[:, :5]
            if len(predlabels) == 0:
                predlabels = predlabel
            else:
                predlabels = np.column_stack((predlabels, predlabel))
            print(predlabels.shape)
            K.clear_session()

        with open("result/" + cur_time + ".csv", 'w') as f:
            for i in range(predlabels.shape[0]):
                # f.write(ques_ids[i] + "," + ','.join([topic_dict_inv[k] for k in predlabels[i]]) + '\n')
                f.write(ques_ids[i] + "," + ','.join(tmpfunc(predlabels[i])) + '\n')

    def save_model(self):
        import time
        cur_time = time.strftime('%Y-%m-%d-%H', time.localtime(time.time()))
        self.model.save(self.savedir + "/latest_twomodel_wordchar_" + str(cur_time) + '.h5')
