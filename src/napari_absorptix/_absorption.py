import numpy as np
import matplotlib.pyplot as plt
from napari.layers import Image, Shapes
from magicgui import magic_factory
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit


def func(x, a, b):
    """
    Linear regression y = a*x + b
    """
    return a * x + b


@magic_factory(call_button='Compute absorption auto')
def compute_absorption_auto(image: Image, rectangle: Shapes) -> None:
    fig, ax = plt.subplots()
    depth = int(image.position[0])
    a, b = butter(4, 0.1, btype='low')
    for data in rectangle.data:
        if data.shape[1] == 3:
            data = data[0][:, 1:]
        y1 = int(np.min(data[:, 0]))
        x1 = int(np.min(data[:, 1]))
        y2 = int(np.max(data[:, 0]))
        x2 = int(np.max(data[:, 1]))
        crop = image.data[depth, y1:y2, x1:x2]

        l = np.log(np.mean(crop, axis=0))
        l1 = filtfilt(a, b, l)
        ind_max = np.argmax(l1)
        l1 = l1[:ind_max]
        x = np.arange(l1.size)
        popt, _ = curve_fit(func, x, l1)
        ax.plot(l, ':')
        ax.plot(l1)
        ax.plot(x, func(x, *popt), label='a={:.2f}'.format(popt[0]*1000))

    plt.legend()
    plt.show()

@magic_factory(call_button='Compute absorption manual')
def compute_absorption_manual(image: Image, rectangle: Shapes) -> None:

    if len(rectangle.data) != 1:
        raise ValueError('Error: this function can only be used with 1 rectangle, got {}.'.format(len(rectangle.data)))

    if rectangle.shape_type[0] != 'rectangle':
        raise TypeError('Error: this function can only be used with rectangle, got {}.'.format(len(rectangle.shape_type)))

    depth = int(image.position[0])
    data = rectangle.data[0]
    if data.shape[1] == 3:
        data = data[:, 1:]
    y1 = int(np.min(data[:, 0]))
    x1 = int(np.min(data[:, 1]))
    y2 = int(np.max(data[:, 0]))
    x2 = int(np.max(data[:, 1]))
    crop = image.data[depth, y1:y2, x1:x2]

    l = np.log(np.mean(crop, axis=0))

    plt.plot(l)
    pts = plt.ginput(n=-1, timeout=-1)
    plt.close()
    pts = np.array(pts)

    print(pts)

    if pts.shape[0] % 2 != 0:
        raise ValueError('The number of points should be even.')

    xs = pts[:, 0].astype(int)
    fig, ax = plt.subplots()
    ax.plot(l)
    for i in range(int(pts.shape[0]/2)):
        x1 = xs[2*i]
        x2 = xs[2*i + 1]

        x = np.arange(x1, x2)
        popt, _ = curve_fit(func, x, l[x1:x2])
        ax.plot(x, func(x, *popt), label='a={:.2f}'.format(popt[0]*1000))

    plt.legend()
    plt.show()