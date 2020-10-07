# Events is a 2d array where the first column contains the sample index when the event occurred.
# The second column contains the value of the trigger channel immediately before the event occurred.
# The third column contains the event-id.
import mne
import os.path as op

from EEG_Processing import exp_func_all
import matplotlib.pyplot as plt

folder_path2 = 'C:\\Users\Estella\Documents\errp_Data\\7. ErrPc1\s2\\all'
raw2 = mne.io.read_raw_edf(op.join(folder_path2, 'e7.20110718.101723.offline.errp.errpSquaresLeft.gdf'), preload=True)

ep2 = exp_func_all(raw2)
# ep2.events
# array([[ 1317,     0,     1],
# #        [ 2002,     0,     2],
# #        [ 2578,     0,     2],
# #        [ 3043,     0,     2],
# #        [ 3699,     0,     2],
# #        [ 6146,     0,     2],
# #        [ 6754,     0,     2],
# #        [ 7443,     0,     2],
# #        [ 9634,     0,     2],
# #        [10258,     0,     2],
# #        [12130,     0,     2],
# #        [12658,     0,     1])

plt.plot(ep2._data[0,0:3,:].T)
plt.plot(ep2._data[99,16,:].T)
plt.plot(ep2._data[0,16,:].T)

plt.plot(ep2._data[:,16,:].T) # Response channel, #17
plt.xlabel('time(s)')
plt.ylabel('EEG channels')



############################################# Fancy Plots ###################################################
mne.viz.plot_events(ep2.events)
ep2.plot(scalings={'eeg': 10}, event_colors={1: 'red', 2: 'blue'})  # Why color for diff event_id not visible?

evoked = ep2.average()
fig = evoked.plot()  # butterfly plots

picks = mne.pick_types(raw2.info, eeg=True)
start, stop = raw2.time_as_index([100, 115])  # 100 s to 115 s data segment
data, times = raw2[picks[:10], start:stop]

import matplotlib.pyplot as plt

plt.plot(times, data.T)
plt.xlabel('time (s)')
plt.ylabel('EEG data (T)')

# Plot cov matrix
noise_cov = mne.compute_covariance(ep2, tmax=0.)
mne.viz.plot_cov(noise_cov, raw2.info)

# Topo map
# evoked.plot_topomap(times=np.linspace(0.05, 0.15, 5), ch_type='eeg')
# RuntimeError: No digitization points found.

# Compute inverse solution: get the brain back.


