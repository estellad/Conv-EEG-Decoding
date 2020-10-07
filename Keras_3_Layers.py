import os
os.environ['PYTHONHASHSEED'] = '0'

import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.set_random_seed(2)
import random
random.seed(9001)


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GaussianDropout, GaussianNoise, Dropout, AlphaDropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras import regularizers
from keras import initializers
import keras.backend as K


# parameters
# Special first block separate convolution(temporal) and spatial.
def special_first_block(model):
    model.add(GaussianNoise(0.012))
    # layer 1: over time
    model.add(Conv2D(filters=25, kernel_size=(1, 3), strides=1, use_bias=False, input_shape=(8, 52, 1),
                     kernel_regularizer=regularizers.l2(0.021),
                     kernel_initializer=initializers.glorot_uniform(2),                          # this is the default.
                     name='conv_time'))
    # layer 2: over spatial
    model.add(Conv2D(filters=25, kernel_size=(8, 1), strides=1, use_bias=False,
                     kernel_regularizer=regularizers.l2(0.024),
                     kernel_initializer=initializers.glorot_uniform(2),                          # this is the default.
                     name='conv_spat'))

    model.add(BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5, name='bnorm'))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3), name="pooling"))
    return model


# Standard 3 blocks. Check
def standard_conv_maxp_block_conv2(model, n_filters):
    suffix = '_{:d}'.format(2)
    model.add(GaussianDropout(0.219, name='drop' + suffix))                               # Current
    # model.add(Dropout(0., seed=1))
    # model.add(AlphaDropout(0.5, seed=1))
    model.add(Conv2D(filters=n_filters, kernel_size=(1, 3), strides=1, use_bias=False,
                     kernel_initializer=initializers.glorot_uniform(2),                          # this is the default.
                     name='conv' + suffix))
    model.add(BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5, name='bnorm' + suffix))

    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3), name='pool' + suffix))
    return model

# Standard 3 blocks. Check
def standard_conv_maxp_block_conv3(model, n_filters):
    suffix = '_{:d}'.format(3)
    model.add(GaussianDropout(0.256, name='drop' + suffix))                                # Current
    # model.add(Dropout(0.5, seed=1))
    # model.add(AlphaDropout(0.5, seed=1))
    model.add(Conv2D(filters=n_filters, kernel_size=(1, 3), strides=1, use_bias=False,
                     kernel_initializer=initializers.glorot_uniform(2),                          # this is the default.
                     name='conv' + suffix))
    model.add(BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5, name='bnorm' + suffix))

    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2), name='pool' + suffix))
    return model


def last_layer(model, num_classes):
    model.add(Flatten())
    model.add(Dense(num_classes, activation='sigmoid',
                    kernel_initializer=initializers.glorot_uniform(2),                           # this is the default.
                    kernel_regularizer=regularizers.l2(0.016)))
    return model


#################################################### AUC #######################################################
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.stack([binary_PFA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
    binSizes = -(pfas[1:] - pfas[:-1])
    s = ptas * binSizes
    return K.sum(s, axis=0)


# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N


# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P
################################################### AUC #########################################################


def Deep3Net(num_classes, n_filters_2, n_filters_3):
    model = Sequential()
    model = special_first_block(model)
    model = standard_conv_maxp_block_conv2(model, n_filters_2)  # n_filters_2 = 50
    model = standard_conv_maxp_block_conv3(model, n_filters_3)  # n_filters_3 = 100
    last_layer(model, num_classes)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[auc])
    return model


# model = Deep3Net(num_classes=1, n_filters_2=70, n_filters_3=80)


######  To big for-loop  ############
from sklearn.metrics import roc_auc_score
# from EEG_Processing import data_collectionC1S6 #, data_collectionC1S3, validation_collectionC2S3, test_collectionC2S3
from sklearn.model_selection import train_test_split

# For one subject under one protocol
def data(data_collection):
    response = data_collection['response']
    data = data_collection['data']

    X_train_val, X_test, y_train_val, y_test = train_test_split(data, response, stratify=response, test_size=0.10, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, stratify=y_train_val, test_size=0.10, random_state=1)

    return X_train, y_train, X_val, y_val, X_test, y_test

#
# data, response, val_data, val_response, test_data, test_response = data(data_collectionC1S6)
# class_weights = {0: len(response[response==1])/len(response[response==0]), 1: 1}
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=0.000001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=30, verbose=0, mode='min')
#
#
# hist = model.fit(data, response, batch_size=128, epochs=100, verbose=1, class_weight=class_weights,
#                  callbacks=[reduce_lr, early_stop], shuffle=True, validation_data=(val_data, val_response))
# y_pred = model.predict(test_data, batch_size=128)
# y_pred_auc = roc_auc_score(test_response, y_pred)
#
# print("Test data AUC:")
# print(y_pred_auc)
