import mne
import numpy as np
import os.path as op
import matplotlib.pyplot as plt


# Location
folder_path = 'C:\\Users\Estella\Documents\errp_Data\\7. ErrPc1\s1\\all'
raw = mne.io.read_raw_edf(op.join(folder_path, 'e8.20110715.150159.offline.errp.errpSquaresRight.gdf'), preload=True)

# Set EEG average reference
raw.set_eeg_reference('average', projection=True)           #CAR

# Return n_channels * n_times, one line info
raw

# Return a list of info, with extraction of channel names
raw.info['ch_names']

# plot Time vs. Channel
# raw.plot(block=True, lowpass=40)

# Plot one single channel of Status
plt.plot(raw._data[16,:])


####################### Epoch the data, and remove bad epochs. #######################
# Extract events, for epoching.
events = mne.find_events(raw, stim_channel='Status')
# 100 events found
# Event IDs: [1 2]

event_id = {'Error': 1, 'Correct': 2}

epochs = mne.Epochs(raw, events, event_id, tmin=-0.1, tmax=1,
                    baseline=(None, 0), preload=True)
print(epochs)

# <Epochs  |   100 events (all good), -0.101562 - 1 sec, baseline [None, 0], ~3.9 MB, data loaded,
#  'Correct': 84
#  'Error': 16>                         # Highly imbalance, don't adjust, according to Fumi?


############################# Competition Dataset ##############################
raw_comp = mne.io.read_raw_edf(op.join('C:\\Users\Estella\Downloads\\BCICIV_2a_gdf', 'A06E.gdf'), preload=True)





