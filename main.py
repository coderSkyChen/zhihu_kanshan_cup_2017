# coding=utf-8
from model import *
import config
from load_data import *
import sys

def results_weight_sum(filenames, topic_dict_inv, ques_ids, weights):
    '''
    ensemble model by weighting sum
    :param filenames: 
    :param topic_dict_inv: 
    :param ques_ids: 
    :param weights: 
    :return: 
    '''
    assert len(weights) == len(filenames)
    import time
    import numpy as np
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

    for i in range(len(filenames)):
        print('process %d th...' % (i))
        predlabel = np.load(filenames[i])
        if len(predlabels) == 0:
            predlabels = predlabel * weights[0]
        else:
            predlabels = predlabels + predlabel * weights[i]
        print(predlabels.shape)

    predlabels = np.argsort(-predlabels)[:, :5]

    with open("final_423.csv", 'w') as f:
        for i in range(predlabels.shape[0]):
            # f.write(ques_ids[i] + "," + ','.join([topic_dict_inv[k] for k in predlabels[i]]) + '\n')
            f.write(ques_ids[i] + "," + ','.join(tmpfunc(predlabels[i])) + '\n')


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('error, give me mode ')
        exit()

    mode = sys.argv[1]

    print(config.get_current_time(), 'current mode:', mode)

    if mode == "train":

        save_root_dir = './model_exp'  #your own path, to save models,tokenizers...

        dl = data_loader(save_root_dir)
        datatuple = dl.load_train_data()
        dl.load_charembedding_matrix()
        dl.load_wordembedding_matrix()

        mymodel = MultiModel(w_embed_matrix=dl.embedword_matrix, c_embed_matrix=dl.embedchar_matrix,
                             word_index=dl.word_index, char_index=dl.char_index, titlechar_length=dl.tc_len,
                             titleword_length=dl.tw_len, dsp_padlen=dl.dsppad_length, data=datatuple,
                             savedir=save_root_dir)
        mymodel.trainmodel(isalldata=True)

    if mode == "pred":
        save_root_dir = './model_4rcnn_att_titledsp_lstm512_lr0_0001_nofc_alldata'   #your own model path
        dl = data_loader(save_root_dir)
        datatuple = dl.load_pred_data_4part()

        mymodel = MultiModel()
        mymodel.predmodel([save_root_dir + "/2017-08-10-12-20_model-06.hdf5"], datatuple=datatuple,
                          topic_dict_inv=dl.topic_dict_inv)

