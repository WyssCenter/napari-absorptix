import os.path
import numpy as np
from typing import List, Any, Optional, Union, Tuple, Dict
from napari.types import ReaderFunction
from napari_plugin_engine import napari_hook_implementation
from napari.layers import Shapes

@napari_hook_implementation
def napari_get_reader(path: Union[str, List[str]]) -> Optional[ReaderFunction]:
    """
    Return a function capable of reading raw files lazily into napari layer data.

    Parameters
    ----------
    path : str or list of str
        Path to a '.raw' file, or a list of such paths.
    Returns
    -------
    function or None
        If the path is of the correct format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
        Otherwise returns ``None``.
    """
    if isinstance(path, list):
        for p in path:
            if not isinstance(p, str) or p.endswith('.raw'):
                return None
    elif isinstance(path, str):
        if not path.endswith('.raw'):
            return None

    return raw_reader


def raw_reader(path: Union[str, List[str]]) -> List[Tuple[Any, Dict, str]]:
    """
    Take a path or list of paths to '.raw' files and return a list of LayerData tuples, with one layer
    for each raw file.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a lazy_raw, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of layer
        (currently always 'image').
    """

    if isinstance(path, str):
        path = [path]

    filenames = []
    for p in path:
        _, filename = os.path.split(p)
        filenames.append(filename)

    reader_list = []
    for p, f in zip(path, filenames):
        reader_list.append((lazy_raw(p), {'name': f, 'contrast_limits': [0, 5000]}, 'image'))

    reader_list.append((None, {'name': 'Draw a rectangle'}, 'shapes'))

    return reader_list


class lazy_raw():
    """
    Class to handle raw binary files lazily

    """
    def __init__(self, path):
        """
        path: str
            path for target data
        """
        self.path = path
        self.dtype = np.uint16

    def __getitem__(self, item):
        if isinstance(item, int):
            return np.fromfile(self.path, count=2048 * 2048, offset=2*item * 2048 * 2048, dtype=self.dtype).reshape(2048, 2048)
        elif isinstance(item, slice):
            return np.fromfile(self.path, count=2048*2048, offset=2*item*2048*2048, dtype=self.dtype).reshape(2048, 2048)
        elif isinstance(item, tuple):
            depth = item[0]
            slice_y = item[1]
            slice_x = item[2]
            y1 = slice_y.start if slice_y.start else 0
            y2 = slice_y.stop if slice_y.stop else 2048
            ny = y2 - y1
            u = np.fromfile(self.path, count=ny * 2048,
                            offset=2*depth * 2048 * 2048 + 2 * y1 * 2048, dtype=self.dtype).reshape(ny, 2048)
            return u[:, slice_x.start:slice_x.stop]
        else:
            return np.fromfile(self.path, count=-1, dtype=self.dtype).reshape(-1, 2048, 2048)

    @property
    def shape(self):
        n_planes = int(os.path.getsize(self.path) / (2*2048*2048))
        return n_planes, 2048, 2048

    @property
    def ndim(self):
        return 3

    def min(self):
        return 0

    def max(self):
        return 5000
