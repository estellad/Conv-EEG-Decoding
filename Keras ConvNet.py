from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ReduceLROnPlateau

# idea first and then class wrap

# parameters

# Special first block separate convolution(temporal) and spatial.
def special_first_block(model, input_tensor, n_filters_time, n_filters_spat, bn_axis, pool_stride,
                        pool_time_length, filter_time_length, conv_stride, in_chans):
    # layer 1: over time
    model.add(Conv2D(filters=n_filters_time, kernel_size=(filter_time_length, 1), input_tensor=input_tensor,
                     use_bias=False, stride=1, name='conv_time'))
    # layer 2: over spatial
    model.add(Conv2D(filters=n_filters_spat, kernel_size=(1, in_chans), stride=(conv_stride, 1), use_bias=False,
                     name='conv_spat'))

    model.add(BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=1e-5, name='bnorm'))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(kernel_size=(pool_time_length, 1), stride=(pool_stride, 1), name="pooling"))


# Standard 3 blocks.
def standard_conv_maxp_block(model, n_filters, filter_length, block_nr, dropout_rate, bn_axis,
                             conv_stride, pool_stride, pool_time_length):
    suffix = '_{:d}'.format(block_nr)
    model.add(Dropout(dropout_rate, name='drop' + suffix))
    model.add(Conv2D(filters=n_filters, kernel_size=(filter_length, 1), strides=(conv_stride, 1), use_bias=False,        # kernel size = (10,1) # stride size(1,1)
                     name='conv' + suffix.format(block_nr)))
    model.add(BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=1e-5, name='bnorm' + suffix))

    model.add(Activation('elu'))
    model.add(MaxPooling2D(kernel_size=(pool_time_length, 1), stride=(pool_stride, 1), name='pool' + suffix))            # Maxpooling kernel size(3,1),  stride (3,1)


def last_layer(model, final_conv_length, num_classes):
    model.add(Conv2D(kernel_size=(final_conv_length, 1), use_bias=False, name='conv_classifier'))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))


def Deep4Net(channels = 16,
             num_classes = 2,
             n_filters_time = 25,  # WTF are these two.
             n_filters_spat = 25,  #
                                   # Number of filters  = number of elu units
                                   # Filter length = stride in CNN?
             filter_time_length = 10,
             dropout_rate = 0.5,
             n_filters_2 = 50,   #
             filter_length_2 = 10,  #
             n_filters_3 = 100,  #
             filter_length_3 = 10,  #
             n_filters_4 = 200,  #
             filter_length_4 = 10,  #
             pool_time_length = 3,  # ? What is length.
             pool_stride = 3,  # All the max pool has been stride 3*1
             input_tensor = (16, 283, 1),
             conv_stride = 1,
             bn_axis = 3, # because Tensorflow backend channel_last.
             final_conv_length = 3
             ):

    model = Sequential()
    special_first_block(model, input_tensor, n_filters_time, n_filters_spat, conv_stride, channels, bn_axis)
    standard_conv_maxp_block(model, n_filters_2, filter_length_2, 2, dropout_rate, bn_axis)        # n_filters_2 = 50,  #filter_length_2 = 10
    standard_conv_maxp_block(model, n_filters_3, filter_length_3, 3, dropout_rate, bn_axis)        # n_filters_3 = 100,  #filter_length_3 = 10
    standard_conv_maxp_block(model, n_filters_4, filter_length_4, 4, dropout_rate, bn_axis)        # n_filters_4 = 200,  #filter_length_4 = 10
    last_layer(model, final_conv_length, num_classes)
    return model


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# What data to throw?!
# Input data as a 2-D array, number of time steps as the width, and the channels as the height.
# decrease learning rate while learning:
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)
hist = model.fit(data, response, batch_size=48, epochs=100, verbose=1, callbacks=[reduce_lr],
                 shuffle=True, validation_split=0.2)

########################################### Nothing shows on the plot ########################################
# data[0].shape = (283, 16, 1)
from PIL import Image
img = Image.fromarray(data1[0], 'RGB')
img.save('my.png')
img.show()

################# Image data generator experiment ##################
