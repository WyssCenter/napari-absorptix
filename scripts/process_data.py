import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

data = np.load('../data/647.npy', allow_pickle=True)

normalize = False
transverse_resolution = 5.26

b, a = butter(3, 0.025, 'lowpass')
fig, ax = plt.subplots(3, 1, sharex=True)
for l in data:
    ind_max = np.argmax(np.diff(filtfilt(b, a, l)))
    if normalize:
        l = l / l.max()

    ax[0].plot(np.arange(-ind_max, l.size - ind_max) * transverse_resolution, l)
    ax[1].plot(np.arange(-ind_max, l.size - ind_max) * transverse_resolution, filtfilt(b, a, l))
    ax[2].plot(np.arange(-ind_max, l.size - ind_max - 1) * transverse_resolution, -np.diff(filtfilt(b, a, l)))

ax[2].set_xlabel('Distance [$\mu m$]')
ax[0].set_ylabel('Intensity (log) [a.u.]')
ax[1].set_ylabel('Intensity (log) filtered [a.u.]')
ax[2].set_ylabel('Intensity (log) derivative [a.u.]')
ax[0].legend(['0h', '3h', '6h'])
ax[1].legend(['0h', '3h', '6h'])
ax[2].legend(['0h', '3h', '6h'])