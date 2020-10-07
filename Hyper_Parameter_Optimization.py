from __future__ import print_function
import os
os.environ['PYTHONHASHSEED'] = '0'

import numpy as np
np.random.seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import random
random.seed(9001)

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GaussianDropout, GaussianNoise, Dropout, AlphaDropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras import regularizers
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from EEG_Processing import data_collectionC1S6 #, data_collectionC1S1, data_collectionC1S2, data_collectionC1S3, data_collectionC1S4, data_collectionC1S5
from EEG_Processing import validation_collectionC2S6 #, validation_collectionC2S1, validation_collectionC2S2, validation_collectionC2S3, validation_collectionC2S4, validation_collectionC2S5
from EEG_Processing import test_collectionC2S6 #, test_collectionC2S2, test_collectionC2S3, test_collectionC2S4, test_collectionC2S5, test_collectionC2S1


from Keras_3_Layers import auc

# for one individual data_C1S6
def data():
    response = data_collectionC1S6['response']
    data = data_collectionC1S6['data']

    X_train_val, X_test, y_train_val, y_test = train_test_split(data, response, stratify=response, test_size=0.10, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, stratify=y_train_val, test_size=0.10, random_state=1)

    return X_train, y_train, X_val, y_val, X_test, y_test


# # for transfer C1S1->C2S1
# def data():
#     X_train = data_collectionC1S6['data']
#     y_train = data_collectionC1S6['response']
#     X_val = validation_collectionC2S6['data']
#     y_val = validation_collectionC2S6['response']
#     X_test = test_collectionC2S6['data']
#     y_test = test_collectionC2S6['response']
#     return X_train, y_train, X_val, y_val, X_test, y_test


############################################### Network with Hyperas Spaces ###########################################
def special_first_block(model):
    model.add(GaussianNoise({{uniform(0, 0.1)}}))
    # layer 1: over time
    model.add(Conv2D(filters={{choice([10, 25, 50])}}, kernel_size=(1, 3), strides=1, use_bias=False, input_shape=(8, 52, 1),
                     kernel_regularizer=regularizers.l2({{uniform(0, 0.05)}}),
                     name='conv_time'))
    # layer 2: over spatial
    model.add(Conv2D(filters={{choice([10, 25, 50])}}, kernel_size=(8, 1), strides=1, use_bias=False,
                     kernel_regularizer=regularizers.l2({{uniform(0, 0.05)}}),
                     name='conv_spat'))

    model.add(BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5, name='bnorm'))
    model.add(Activation({{choice(['relu', 'elu', 'selu'])}}))
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3), name="pooling"))
    return model


def standard_conv_maxp_block_conv2(model, n_filters):
    suffix = '_{:d}'.format(2)
    # model.add({{choice([Dropout(0.5), GaussianDropout(0.02), AlphaDropout(0.02)])}}, name='drop' + suffix)
    # model.add(Dropout({{uniform(0, 1)}}))
    model.add(GaussianDropout({{uniform(0, 1)}}))
    # model.add(AlphaDropout({{uniform(0, 1)}}))
    model.add(Conv2D(filters=n_filters, kernel_size=(1, 3), strides=1, use_bias=False,
                     name='conv' + suffix))
    model.add(BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5, name='bnorm' + suffix))

    model.add(Activation({{choice(['elu', 'selu'])}}))
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3), name='pool' + suffix))
    return model


def standard_conv_maxp_block_conv3(model, n_filters):
    suffix = '_{:d}'.format(3)
    # model.add({{choice([Dropout(0.5), GaussianDropout(0.02), AlphaDropout(0.02)])}}, name='drop' + suffix)
    # model.add(Dropout({{uniform(0, 1)}}))
    model.add(GaussianDropout({{uniform(0, 1)}}))
    # model.add(AlphaDropout({{uniform(0, 1)}}))
    model.add(Conv2D(filters=n_filters, kernel_size=(1, 3), strides=1, use_bias=False,
                     name='conv' + suffix))
    model.add(BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5, name='bnorm' + suffix))

    model.add(Activation({{choice(['elu', 'selu'])}}))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2), name='pool' + suffix))
    return model


def last_layer(model, num_classes):
    model.add(Flatten())
    model.add(Dense(num_classes, activation='sigmoid', kernel_regularizer=regularizers.l2({{uniform(0, 0.1)}})))
    return model


def Deep3Net(num_classes):
    model = Sequential()
    model = special_first_block(model)
    model = standard_conv_maxp_block_conv2(model, n_filters={{choice([30, 50, 70])}})  # n_filters_2 = 50
    model = standard_conv_maxp_block_conv3(model, n_filters={{choice([80, 100, 120])}})  # n_filters_3 = 100
    last_layer(model, num_classes)
    model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}}, loss='binary_crossentropy', metrics=[auc])
    return model
################################################# Network Helpers End ##############################################


def create_model(X_train, y_train, X_val, y_val, X_test, y_test):
    model = Deep3Net(num_classes=1)
    class_weights = {0: len(y_train[y_train == 1]) / len(y_train[y_train == 0]), 1: 1}
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=0.000001)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=30, verbose=1, mode='min')

    model.fit(X_train, y_train, batch_size={{choice([64, 128])}}, epochs=100, verbose=0, class_weight=class_weights,
              callbacks=[reduce_lr, early_stop], shuffle=True, validation_data=(X_val, y_val))

    y_pred = model.predict(X_test, batch_size=128)
    y_pred_auc = roc_auc_score(y_test, y_pred)
    print('Test AUC:', y_pred_auc)
    return {'loss': -y_pred_auc, 'status': STATUS_OK, 'model': model}


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = data()
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=50,
                                          trials=Trials(),
                                          eval_space=True)

    print("Evalutation of best performing model:")
    print('best_test_auc =' + str(roc_auc_score(y_test, best_model.predict(X_test, batch_size=128))))

    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    # best_model.save('best_modelC1S1.h5')


if __name__ == '__main__':
    main()
