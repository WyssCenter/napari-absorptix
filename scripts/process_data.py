import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, correlate

Lambda = 647
exp = 'parafin'
data = list(np.load('../data/{}/{}.npy'.format(exp, Lambda), allow_pickle=True))
# data.extend([l for l in np.load('../data/{}/{}_2.npy'.format(exp, Lambda), allow_pickle=True)])

normalize = False
transverse_resolution = 5.26

b, a = butter(3, 0.025, 'lowpass')
b2, a2 = butter(3, 0.01, 'lowpass')
fig, ax = plt.subplots(3, 1, sharex=True)
for i, l in enumerate(data):
    l = l.astype(float)

    if i ==0:
        normf = np.percentile(np.exp(l), 90)

    ind_max = np.argmin(np.diff(filtfilt(b, a, l)))

    if normalize:
        l = l / l.max()

    ax[0].plot(np.arange(-ind_max, l.size - ind_max) * transverse_resolution, l)
    ax[1].plot(np.arange(-ind_max, l.size - ind_max) * transverse_resolution, np.exp(l)/normf)
    ax[2].plot(np.arange(-ind_max, l.size - ind_max - 1) * transverse_resolution, -np.diff(filtfilt(b2, a2, l)))

ax[2].set_xlabel('Distance [$\mu m$]')
ax[0].set_ylabel('Intensity (log) [a.u.]')
ax[1].set_ylabel('Intensity [a.u.]')
ax[2].set_ylabel('Intensity (log) derivative [a.u.]')
time = ['0h', '3h', '6h', '9h', '12h', '24h', '48h']
for i in range(3):
    ax[i].legend(time)


# For napari
# import matplotlib.pyplot as plt
# import numpy as np
#
# ax = plt.gca()
# data = []
# for l in ax.lines:
#     data.append(l.get_ydata())
# data = np.array(data, dtype=object)
# np.save('405_2.npy', data)