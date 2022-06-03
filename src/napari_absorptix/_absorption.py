import numpy as np
import matplotlib.pyplot as plt
from napari.layers import Image, Shapes
from napari.viewer import Viewer
from magicgui import magic_factory
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit

from ._reader import lazy_raw


def func(x, a, b):
    """
    Linear regression y = a*x + b
    """
    return a * x + b


@magic_factory(call_button='Compute absorption auto',
               transverse_resolution={'tooltip': 'Transverse resolution in micrometers.'})
def compute_absorption_auto(image: Image,
                            rectangle: Shapes,
                            transverse_resolution: float=5.26) -> None:
    """
    Compute optical absorption automatically. The function automatically tries to find the portion of the signal to fit
    (basically the sample edge).

    Parameters
    ----------
    image: napari.layers.Image
        Napari layer containing the image to compute the absorption on
    rectangle: napari.layers.Shapes
        Napari layer containing the ROI (must be rectangles) to compute the absorption on.
    transverse_resolution: float
        transverse resolution in micrometers used for scaling absorption and for plotting.

    Returns
    -------
    None
    """
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
        x = np.arange(l1.size)*transverse_resolution
        popt, _ = curve_fit(func, x, l1)
        ax.plot(np.arange(l.size)*transverse_resolution, l, ':')
        ax.plot(x, l1)
        ax.plot(x, func(x, *popt), label='a={:.2f}'.format(popt[0]*1000))
        ax.set_xlabel('Distance [$\mu m$]')
        ax.set_ylabel('Intensity (log) [a.u.]')

    plt.legend()
    plt.show()

@magic_factory(call_button='Compute absorption manual',
               transverse_resolution={'tooltip': 'Transverse resolution in micrometers.'})
def compute_absorption_manual(image: Image,
                              rectangle: Shapes,
                              transverse_resolution: float=5.26) -> None:
    """
    Compute optical absorption with manual inputs. For each pairs of points the function will fit the intensity data
    to obtain the absorption parameter in the user giver area.

    Parameters
    ----------
    image: napari.layers.Image
        Napari layer containing the image to compute the absorption on
    rectangle: napari.layers.Shapes
        Napari layer containing the ROI (must be rectangles) to compute the absorption on.
    transverse_resolution: float
        transverse resolution in micrometers used for scaling absorption and for plotting.

    Returns
    -------
    None
    """

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
    plt.xlabel('Distance [$\mu m$]')
    plt.ylabel('Intensity (log) [a.u.]')
    plt.title('CLICK FOR FIT')

    plt.plot(np.arange(l.size)*transverse_resolution, l)
    pts = plt.ginput(n=-1, timeout=-1)
    plt.close()
    pts = np.array(pts)

    if pts.shape[0] % 2 != 0:
        raise ValueError('The number of points should be even.')

    xs = (pts[:, 0]/transverse_resolution).astype(int)
    xs[xs < 0] = 0
    fig, ax = plt.subplots()
    ax.plot(np.arange(l.size)*transverse_resolution, l)
    for i in range(int(pts.shape[0]/2)):
        x1 = xs[2*i]
        x2 = xs[2*i + 1]

        x = np.arange(x1, x2)*transverse_resolution

        popt, _ = curve_fit(func, x, l[x1:x2])
        ax.plot(x, func(x, *popt), label='a={:.2f}'.format(popt[0]*1000))

    ax.set_xlabel('Distance [$\mu m$]')
    ax.set_ylabel('Intensity (log) [a.u.]')

    plt.legend()
    plt.show()

@magic_factory(call_button='Plot profile',
               transverse_resolution={'tooltip': 'Transverse resolution in micrometers.'},
               normalize={                         'tooltip': 'normalize by the maximum of each profile'})
def plot_profile(viewer: Viewer,
                 rectangle: Shapes,
                 transverse_resolution: float=5.26,
                 normalize: bool=False) -> None:

    depth = int(viewer.layers[0].position[0])

    for data in rectangle.data:
        if data.shape[1] == 3:
            data = data[0][:, 1:]

        y1 = int(np.min(data[:, 0]))
        x1 = int(np.min(data[:, 1]))
        y2 = int(np.max(data[:, 0]))
        x2 = int(np.max(data[:, 1]))

        for layer in viewer.layers:
            if isinstance(layer, Image) and isinstance(layer.data, lazy_raw):
                crop = layer.data[depth, y1:y2, x1:x2]
                l = np.log(np.mean(crop, axis=0))
                if normalize:
                    l = l/l.max()
                plt.plot(np.arange(l.size) * transverse_resolution, l, label=layer.name)

        plt.xlabel('Distance [$\mu m$]')
        plt.ylabel('Intensity (log) [a.u.]')
        plt.legend()
        plt.show()