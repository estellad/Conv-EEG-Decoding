# 10 fold CV for E2 E3
import os
os.environ['PYTHONHASHSEED'] = '0'

import numpy as np
np.random.seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import random
random.seed(9001)


from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from Keras_3_Layers import Deep3Net
# for C2
# from EEG_Processing import data_collectionC2S1, data_collectionC2S2, data_collectionC2S3, data_collectionC2S4, data_collectionC2S5, data_collectionC2S6
# for C3
from EEG_Processing import data_collectionC3S1, data_collectionC3S2, data_collectionC3S3, data_collectionC3S4, data_collectionC3S5, data_collectionC3S6


def ten_fold_split_train_val(data_collection):
    response = data_collection['train_response']
    data = data_collection['train_data']

    X_train_val = []
    X_test = []
    y_train_val = []
    y_test = []

    kf_1 = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    for train_val_index, test_index in kf_1.split(data, response):
        X_train_val.append(data[train_val_index])
        X_test.append(data[test_index])
        y_train_val.append(response[train_val_index])
        y_test.append(response[test_index])

    train_val_min = min(map(len, y_train_val))
    test_min = min(map(len, y_test))
    for i in range(10):
        if len(X_train_val[i]) > train_val_min:
            num_longer_j = len(X_train_val[i]) - train_val_min
            for j in range(num_longer_j):
                X_train_val[i] = np.delete(X_train_val[i], -(i + 1), 0)
                y_train_val[i] = np.delete(y_train_val[i], -(i + 1), 0)

        if len(X_test[i]) > test_min:
            num_longer_k = len(X_test[i]) - test_min
            for k in range(num_longer_k):
                X_test[i] = np.delete(X_test[i], -(i + 1), 0)
                y_test[i] = np.delete(y_test[i], -(i + 1), 0)

    X_train_val = np.asarray(X_train_val)
    X_test = np.asarray(X_test)
    y_train_val = np.asarray(y_train_val)
    y_test = np.asarray(y_test)

    X_train = [None]*10
    X_val = [None]*10
    y_train = [None]*10
    y_val = [None]*10
    for i in range(10):
        X_train[i], X_val[i], y_train[i], y_val[i] = train_test_split(X_train_val[i], y_train_val[i], random_state=1,
                                                                      stratify=y_train_val[i], test_size=0.10)

    ten_fold_data_collection = dict(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                                    X_test=X_test, y_test=y_test)
    return ten_fold_data_collection


# ten_fold_data_collection = ten_fold_split_train_val(all_subject_collectionC2)
# ten_fold_data_collection['X_train'].shape ->  (10, 5325, 8, 52, 1)


def model_fit_pred(ten_fold_data_collection, model):
    list_auc = []
    for i in range(10):
        response=ten_fold_data_collection['y_train'][i]
        class_weights = {0: len(response[response==1])/len(response[response==0]), 1: 1}
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=0.000001)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=30, verbose=1, mode='min')

        model.fit(ten_fold_data_collection['X_train'][i], response,
                  batch_size=128, epochs=100, verbose=1,
                  callbacks=[reduce_lr, early_stop], shuffle=True,
                  validation_data=(ten_fold_data_collection['X_val'][i], ten_fold_data_collection['y_val'][i]),
                  class_weight=class_weights)
        y_pred = model.predict(ten_fold_data_collection['X_test'][i], batch_size=128)
        y_pred_auc = roc_auc_score(ten_fold_data_collection['y_test'][i], y_pred)
        list_auc.append(y_pred_auc)
    list_auc.append(np.asarray(list_auc).mean())
    list_auc.append(np.asarray(list_auc).std())
    return list_auc


def data_handling(train, subject):
    train_data = eval('data_collectionC' + str(train) + 'S' + str(subject))['data']
    train_response = eval('data_collectionC' + str(train) + 'S' + str(subject))['response']

    data_collection = dict(train_data=train_data, train_response=train_response)
    return data_collection


model = Deep3Net(1, 70, 80)

############### for C2 ###############
# list_C2 = []
# for i in range(6):
#     list_auc_i = model_fit_pred(ten_fold_split_train_val(data_handling(2, i+1)), model)
#     list_C2.append(list_auc_i)


############# for C3 #################
list_C3 = []
for i in range(6):
    list_auc_i = model_fit_pred(ten_fold_split_train_val(data_handling(3, i+1)), model)
    list_C3.append(list_auc_i)


