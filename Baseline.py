import os
os.environ['PYTHONHASHSEED'] = '0'

import numpy as np
np.random.seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import random
random.seed(9001)

from Keras_3_Layers import Deep3Net, reduce_lr, early_stop
from sklearn.metrics import roc_auc_score

# for C2
# from EEG_Processing import validation_collectionC2S1, validation_collectionC2S2, validation_collectionC2S3, validation_collectionC2S4, validation_collectionC2S5, validation_collectionC2S6
# from EEG_Processing import test_collectionC2S1, test_collectionC2S2, test_collectionC2S3, test_collectionC2S4, test_collectionC2S5, test_collectionC2S6

# for C3
from EEG_Processing import validation_collectionC3S1, validation_collectionC3S2, validation_collectionC3S3, validation_collectionC3S4, validation_collectionC3S5, validation_collectionC3S6
from EEG_Processing import test_collectionC3S1, test_collectionC3S2, test_collectionC3S3, test_collectionC3S4, test_collectionC3S5, test_collectionC3S6

# Data for baseline transfer classifier: train on val, test on test on C2, C3
# val_data = validation_collectionC2S3['data']
# val_response = validation_collectionC2S3['response']
# test_data = test_collectionC2S3['data']
# test_response = test_collectionC2S3['response']

def data_handling_bs(subject, val_test):
    val_data = eval('validation_collectionC' + str(val_test) + 'S' + str(subject))['data']
    val_response = eval('validation_collectionC' + str(val_test) + 'S' + str(subject))['response']

    test_data = eval('test_collectionC' + str(val_test) + 'S' + str(subject))['data']
    test_response = eval('test_collectionC' + str(val_test) + 'S' + str(subject))['response']
    class_weights = {0: len(val_response[val_response == 1])/len(val_response[val_response == 0]), 1: 1}

    data_collection = dict(train_data=val_data, train_response=val_response,
                           test_data=test_data, test_response=test_response,
                           class_weights=class_weights)
    return data_collection


def model_fit_pred(model_net, subject, val_test):
    data_collection = data_handling_bs(subject, val_test)
    model_net.fit(data_collection['train_data'], data_collection['train_response'],
                  batch_size=128, epochs=100, verbose=1,
                  callbacks=[reduce_lr, early_stop], shuffle=False,
                  validation_split=0.2,
                  class_weight=data_collection['class_weights'])

    y_pred = model_net.predict(data_collection['test_data'], batch_size=128)
    y_pred_auc = roc_auc_score(data_collection['test_response'], y_pred)
    return y_pred_auc


def for_loop(val_test_protocol):
    list_subjects = ['1', '2', '3', '4', '5', '6']
    model_net = Deep3Net(num_classes=1, n_filters_2=70, n_filters_3=80)
    list_auc = []
    for i in list_subjects:
        y_pred_auc = model_fit_pred(model_net, i, val_test_protocol)
        list_auc.append(y_pred_auc)
    return list_auc

################# for C2 ###################
# c2_bs_result_five = []
# for i in range(5):
#     c2_list = for_loop(2)
#     c2_bs_result_five.append(c2_list)

################# for C3 ###################
c3_bs_result_five = []
for i in range(5):
    c3_list = for_loop(3)
    c3_bs_result_five.append(c3_list)
