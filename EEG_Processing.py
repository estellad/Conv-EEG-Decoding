import os
os.environ['PYTHONHASHSEED'] = '0'

import numpy as np
np.random.seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import random
random.seed(9001)

import mne, glob


# Location
Ctrl1 = 'C:\\Users\cnbiuser\Desktop\errp_Data\\7. ErrPc1'  # Correct 3022, # Error: 1039
Ctrl2 = 'C:\\Users\cnbiuser\Desktop\errp_Data\\8. ErrPc2'  # Correct 3446, # Error: 1827
Ctrl3 = 'C:\\Users\cnbiuser\Desktop\errp_Data\\9. ErrPc3'  # Correct 3498, # Error: 2012 # with three files corrected!


########################################### Helpers ########################################
# Set EEG average reference
def spatial_filter(raw_dat):
    spatial_filtered_dat = raw_dat.set_eeg_reference('average')  # , projection=True)  # CAR
    return spatial_filtered_dat


def spectral_filter(spatial_filtered_dat):
    spectral_filtered_dat = spatial_filtered_dat.filter(l_freq=1, h_freq=10, method='iir')
    spectral_filtered_dat.notch_filter(50)
    spectral_filtered_dat.info['lowpass'] = 10
    spectral_filtered_dat.info['highpass'] = 1
    return spectral_filtered_dat


def epoch_data(filtered_dat, data_type):
    '''
    If data_type is for training, then we will augment the data for the minority class.
    Otherwise, we will just perform normal epoch for the imbalanced data, without augmentation.

    :param filtered_dat: After all filter steps finished, the data that is ready to be epoched.
    :param data_type: one of 'validation_data', 'test_data' or None (for training data)
    :return:
    '''
    if np.unique(filtered_dat._data[8, :])[1] == 16 and np.unique(filtered_dat._data[8, :])[2] == 32:
        filtered_dat._data[8, :][filtered_dat._data[8, :] == 16] = 1
        filtered_dat._data[8, :][filtered_dat._data[8, :] == 32] = 2

    events = mne.find_events(filtered_dat, stim_channel='Status')
    event_id = {'Error': 1, 'Correct': 2}

    if (data_type == 'validation_data') or (data_type == 'test_data'):
        epochs = mne.Epochs(filtered_dat, events, event_id, tmin=0.2, tmax=1.0, baseline=None, preload=True)
        epochs = epochs.resample(64, 'auto')

    elif data_type is None:
        error_events = events[events[:, 2] == 1]                                              #minority class
        correct_events = events[events[:, 2] == 2]
        multiplicative_diff = int(np.floor(len(correct_events)/len(error_events)))
        epochs_0 = mne.Epochs(filtered_dat, events, event_id, tmin=0.2, tmax=1.0, baseline=None, preload=True)
        epochs_0 = epochs_0.resample(64, 'auto')

        list_err_epochs = []
        event_id_err = {'Error': 1}
        for i in range(multiplicative_diff-1):
            mat = np.zeros((len(events[events[:, 2] == 1]), 3))
            random.seed(9001)
            a = np.asarray(random.sample(range(-25, 0), int(len(events[events[:, 2] == 1])/2)))
            random.seed(9001)
            b = np.asarray(random.sample(range(1, 26), len(events[events[:, 2] == 1]) - len(a)))
            random.seed(9001)
            mat[:, 0] = np.asarray(list(map(next, random.sample([iter(a)]*len(a) + [iter(b)]*len(b), len(a)+len(b)))))

            event_i = events[events[:, 2] == 1] + mat.astype('int64')
            epochs_i = mne.Epochs(filtered_dat, event_i, event_id_err, baseline=None, tmin=0.2, tmax=1.0, preload=True)
            list_err_epochs.append(epochs_i.resample(64, 'auto'))

        list_err_epochs.append(epochs_0)
        epochs = mne.concatenate_epochs(list_err_epochs)

    return epochs


def data_response_prep(ep):
    data = np.delete(ep._data, 8, 1)
    data = np.expand_dims(data, axis=3) #ex. (4682, 8, 205, 1)
    response = ep.events[:, 2] - 1
    data_collection = dict(data=data, response=response)
    return data_collection


# This is for combining all subjects' files under one design. # Next Step.
def integrate_all_subject_files(path, data_type):
    '''This function is useless for my current experiment,
    but it is useful for looping and combining all files under all subjects, under one protocol.

       path: one of the 3 paths specified above.
       data_type: one of 'validation_data', 'test_data' or None (for training data)

    '''
    list_epochs = []
    listdir = os.listdir(path)
    for dir in listdir:
        print("Currently under folder" + dir)
        if ".sh" not in dir:
            os.chdir(path + '\\' + dir + "\\all")
            files = glob.glob('*.gdf')

            if data_type == 'validation_data':       # Only take "offline" data
                to_return = []
                for i in files:
                    if 'offline' in i:
                        to_return.append(i)
                files = to_return

            elif data_type == 'test_data':           # Only take "online" data
                to_return = []
                for i in files:
                    if 'online' in i:
                        to_return.append(i)
                files = to_return

            for i in files:
                raw = mne.io.read_raw_edf(i, preload=True)
                raw.pick_types(eeg=True)
                raw.info['chs'][-1]['kind'] = 3
                raw.pick_channels(['eeg:1', 'eeg:3', 'eeg:4', 'eeg:5', 'eeg:8', 'eeg:9', 'eeg:10', 'eeg:14', 'Status'])
                spacial_filtered_dat = spatial_filter(raw)
                filtered_dat = spectral_filter(spacial_filtered_dat)
                epochs = epoch_data(filtered_dat, data_type)
                list_epochs.append(epochs)

    return list_epochs


################################################## Ultimate Process ###################################################
# Read in experiment data for Control/Exp 1, 2, 3
def read_process_combine_data(path, subject, data_type):
    '''

    :param path:
    :param subject: The subject index.
    :param data_type: The 'validation_data' means offline data(maximum 200 trials) Unaugmented;
                      the 'test_data' means online data(max 400 trials). Unaugmented
    :return:
    '''
    if subject is None:
        list_epochs = integrate_all_subject_files(path, data_type)

    # This is for combining subject-wise files.
    else:
        list_epochs = []
        os.chdir(path + '\\' + subject + "\\all")
        files = glob.glob('*.gdf')

        if data_type == 'validation_data':
            to_return = []
            for i in files:
                if 'offline' in i:
                    to_return.append(i)
            files = to_return

        elif data_type == 'test_data':
            to_return = []
            for i in files:
                if 'online' in i:
                    to_return.append(i)
            files = to_return

        for i in files:
            raw = mne.io.read_raw_edf(i, preload=True)
            raw.pick_types(eeg=True)
            raw.info['chs'][-1]['kind'] = 3
            raw.pick_channels(['eeg:1', 'eeg:3', 'eeg:4', 'eeg:5', 'eeg:8', 'eeg:9', 'eeg:10', 'eeg:14', 'Status'])
            spacial_filtered_dat = spatial_filter(raw)
            filtered_dat = spectral_filter(spacial_filtered_dat)
            epochs = epoch_data(filtered_dat, data_type)
            list_epochs.append(epochs)

    concatenated_epochs = mne.concatenate_epochs(list_epochs)
    print(concatenated_epochs)
    data_collection = data_response_prep(concatenated_epochs)

    if data_type == 'validation_data':
        if len(data_collection['data']) > 200:
            data_collection['data'] = data_collection['data'][0:200, :, :, :]
            data_collection['response'] = data_collection['response'][0:200]

    if data_type == 'test_data':
        if len(data_collection['data']) > 400:
            data_collection['data'] = data_collection['data'][0:400, :, :, :]
            data_collection['response'] = data_collection['response'][0:400]

    return data_collection


#######################################################################################################################
# Subject Wise
############################################## Training ###############################################################
# Control
# data_collectionC1S1 = read_process_combine_data(Ctrl1, 's1', None) #C:485 E:126 #7
# data_collectionC2S1 = read_process_combine_data(Ctrl2, 's1', None) #C:499 E:201 #6
data_collectionC3S1 = read_process_combine_data(Ctrl3, 's1', None) #C:519 E:250 #9
#
# data_collectionC1S2 = read_process_combine_data(Ctrl1, 's2', None) #C:458 E:90 #6
# data_collectionC2S2 = read_process_combine_data(Ctrl2, 's2', None) #C:449 E:248 #8
data_collectionC3S2 = read_process_combine_data(Ctrl3, 's2', None) #C:588 E:277 #10
#
# data_collectionC1S3 = read_process_combine_data(Ctrl1, 's3', None) #C:557 E:202 #9
# data_collectionC2S3 = read_process_combine_data(Ctrl2, 's3', None) #C:550 E:350 #9
data_collectionC3S3 = read_process_combine_data(Ctrl3, 's3', None) #C:609 E:491 #11
#
# data_collectionC1S4 = read_process_combine_data(Ctrl1, 's4', None) #C:630 E:270 #9
# data_collectionC2S4 = read_process_combine_data(Ctrl2, 's4', None) #C:745 E:455 #12
data_collectionC3S4 = read_process_combine_data(Ctrl3, 's4', None) #C:485 E:321 #13
#
# data_collectionC1S5 = read_process_combine_data(Ctrl1, 's5', None) #C:443 E:121 #8
# data_collectionC2S5 = read_process_combine_data(Ctrl2, 's5', None) #C:592 E:289 #10
data_collectionC3S5 = read_process_combine_data(Ctrl3, 's5', None) #C:548 E:293 #9
#
# data_collectionC1S6 = read_process_combine_data(Ctrl1, 's6', None) #C:449 E:230 #7
# data_collectionC2S6 = read_process_combine_data(Ctrl2, 's6', None) #C:611 E:284 #10
data_collectionC3S6 = read_process_combine_data(Ctrl3, 's6', None) #C:559 E:380 #10


################################################## Validation ######################################################
# # Control
# # validation_collectionC1S1 = read_process_combine_data(Ctrl1, 's1', 'validation_data')
# # validation_collectionC2S1 = read_process_combine_data(Ctrl2, 's1', 'validation_data')
# validation_collectionC3S1 = read_process_combine_data(Ctrl3, 's1', 'validation_data')
# #
# # validation_collectionC1S2 = read_process_combine_data(Ctrl1, 's2', 'validation_data')
# # validation_collectionC2S2 = read_process_combine_data(Ctrl2, 's2', 'validation_data')
# validation_collectionC3S2 = read_process_combine_data(Ctrl3, 's2', 'validation_data')
# #
# # validation_collectionC1S3 = read_process_combine_data(Ctrl1, 's3', 'validation_data')
# # validation_collectionC2S3 = read_process_combine_data(Ctrl2, 's3', 'validation_data')
# validation_collectionC3S3 = read_process_combine_data(Ctrl3, 's3', 'validation_data')
# #
# # validation_collectionC1S4 = read_process_combine_data(Ctrl1, 's4', 'validation_data')
# # validation_collectionC2S4 = read_process_combine_data(Ctrl2, 's4', 'validation_data')
# validation_collectionC3S4 = read_process_combine_data(Ctrl3, 's4', 'validation_data')
# #
# # validation_collectionC1S5 = read_process_combine_data(Ctrl1, 's5', 'validation_data')
# # validation_collectionC2S5 = read_process_combine_data(Ctrl2, 's5', 'validation_data')
# validation_collectionC3S5 = read_process_combine_data(Ctrl3, 's5', 'validation_data')
# #
# # validation_collectionC1S6 = read_process_combine_data(Ctrl1, 's6', 'validation_data')
# # validation_collectionC2S6 = read_process_combine_data(Ctrl2, 's6', 'validation_data')
# validation_collectionC3S6 = read_process_combine_data(Ctrl3, 's6', 'validation_data')


################################################## Test ######################################################
# # Control
# # test_collectionC1S1 = read_process_combine_data(Ctrl1, 's1', 'test_data')
# # test_collectionC2S1 = read_process_combine_data(Ctrl2, 's1', 'test_data')
# test_collectionC3S1 = read_process_combine_data(Ctrl3, 's1', 'test_data')
# #
# # test_collectionC1S2 = read_process_combine_data(Ctrl1, 's2', 'test_data')
# # test_collectionC2S2 = read_process_combine_data(Ctrl2, 's2', 'test_data')
# test_collectionC3S2 = read_process_combine_data(Ctrl3, 's2', 'test_data')
# #
# # test_collectionC1S3 = read_process_combine_data(Ctrl1, 's3', 'test_data')
# # test_collectionC2S3 = read_process_combine_data(Ctrl2, 's3', 'test_data')
# test_collectionC3S3 = read_process_combine_data(Ctrl3, 's3', 'test_data')
# #
# # test_collectionC1S4 = read_process_combine_data(Ctrl1, 's4', 'test_data')
# # test_collectionC2S4 = read_process_combine_data(Ctrl2, 's4', 'test_data')
# test_collectionC3S4 = read_process_combine_data(Ctrl3, 's4', 'test_data')
# #
# # test_collectionC1S5 = read_process_combine_data(Ctrl1, 's5', 'test_data')
# # test_collectionC2S5 = read_process_combine_data(Ctrl2, 's5', 'test_data')
# test_collectionC3S5 = read_process_combine_data(Ctrl3, 's5', 'test_data')
# #
# # test_collectionC1S6 = read_process_combine_data(Ctrl1, 's6', 'test_data')
# # test_collectionC2S6 = read_process_combine_data(Ctrl2, 's6', 'test_data')
# test_collectionC3S6 = read_process_combine_data(Ctrl3, 's6', 'test_data')



