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
from sklearn.utils import shuffle
# pair 1,2
from EEG_Processing import data_collectionC1S1, data_collectionC1S2, data_collectionC1S3, data_collectionC1S4, data_collectionC1S5, data_collectionC1S6
from EEG_Processing import validation_collectionC2S1, validation_collectionC2S2, validation_collectionC2S3, validation_collectionC2S4, validation_collectionC2S5, validation_collectionC2S6
from EEG_Processing import test_collectionC2S1, test_collectionC2S2, test_collectionC2S3, test_collectionC2S4, test_collectionC2S5, test_collectionC2S6

# # pair 1,3
# from EEG_Processing import data_collectionC1S1, data_collectionC1S2, data_collectionC1S3, data_collectionC1S4, data_collectionC1S5, data_collectionC1S6
# from EEG_Processing import validation_collectionC3S1, validation_collectionC3S2, validation_collectionC3S3, validation_collectionC3S4, validation_collectionC3S5, validation_collectionC3S6
# from EEG_Processing import test_collectionC3S1, test_collectionC3S2, test_collectionC3S3, test_collectionC3S4, test_collectionC3S5, test_collectionC3S6

# # pair 2,3
# from EEG_Processing import data_collectionC2S1, data_collectionC2S2, data_collectionC2S3, data_collectionC2S4, data_collectionC2S5, data_collectionC2S6
# from EEG_Processing import validation_collectionC3S1, validation_collectionC3S2, validation_collectionC3S3, validation_collectionC3S4, validation_collectionC3S5, validation_collectionC3S6
# from EEG_Processing import test_collectionC3S1, test_collectionC3S2, test_collectionC3S3, test_collectionC3S4, test_collectionC3S5, test_collectionC3S6



def data_handling(train, subject, val_test):
    train_data_prep = eval('data_collectionC' + str(train) + 'S' + str(subject))['data']
    train_response_prep = eval('data_collectionC' + str(train) + 'S' + str(subject))['response']
    val_data_prep = eval('validation_collectionC' + str(val_test) + 'S' + str(subject))['data']
    val_response_prep = eval('validation_collectionC' + str(val_test) + 'S' + str(subject))['response']

    mixture_train_data = np.concatenate((train_data_prep, val_data_prep), axis=0)
    mixture_train_response = np.concatenate((train_response_prep, val_response_prep), axis=0)
    train_val_size = len(mixture_train_response)
    train_size = train_val_size - 200

    shuffled_index = shuffle(list(range(train_val_size)), random_state=2)
    train_data = np.asarray([mixture_train_data[n] for n in [shuffled_index[x] for x in list(range(train_size))]])
    val_data = np.asarray([mixture_train_data[n] for n in [shuffled_index[x] for x in list(range(train_size, train_val_size))]])
    train_response = np.asarray([mixture_train_response[n] for n in [shuffled_index[x] for x in list(range(train_size))]])
    val_response = np.asarray([mixture_train_response[n] for n in [shuffled_index[x] for x in list(range(train_size, train_val_size))]])

    test_data = eval('test_collectionC' + str(val_test) + 'S' + str(subject))['data']
    test_response = eval('test_collectionC' + str(val_test) + 'S' + str(subject))['response']
    class_weights = {0: len(train_response[train_response == 1])/len(train_response[train_response == 0]), 1: 1}

    data_collection = dict(train_data=train_data, train_response=train_response, val_data=val_data,
                           val_response=val_response, test_data=test_data, test_response=test_response,
                           class_weights=class_weights)

    return data_collection


def model_fit_pred(model_net, train, subject, val_test):
    data_collection = data_handling(train, subject, val_test)
    model_net.fit(data_collection['train_data'], data_collection['train_response'],
              batch_size=128, epochs=100, verbose=1,
              callbacks=[reduce_lr, early_stop], shuffle=False,
              validation_data=(data_collection['val_data'], data_collection['val_response']),
              class_weight=data_collection['class_weights'])
    y_pred = model_net.predict(data_collection['test_data'], batch_size=128)
    y_pred_auc = roc_auc_score(data_collection['test_response'], y_pred)
    return y_pred_auc


def pair_wise_for_loop(train_protocol, test_protocol):
    list_subjects = ['1', '2', '3', '4', '5', '6']
    model_net = Deep3Net(num_classes=1, n_filters_2=70, n_filters_3=80)
    list_auc = []
    for i in list_subjects:
        y_pred_auc = model_fit_pred(model_net, train_protocol, i, test_protocol)
        list_auc.append(y_pred_auc)
    return list_auc


c1c2_result_five = []
for i in range(5):
    c1c2_list = pair_wise_for_loop(1,2)
    c1c2_result_five.append(c1c2_list)

# c1c3_result_five = []
# for i in range(5):
#     c1c3_list = pair_wise_for_loop(1,3)
#     c1c3_result_five.append(c1c3_list)

# c2c3_result_five = []
# for i in range(5):
#     c2c3_list = pair_wise_for_loop(2,3)
#     c2c3_result_five.append(c2c3_list)
