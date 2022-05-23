from ._absorption import compute_absorption_auto, compute_absorption_manual

def napari_experimental_provide_dock_widget():
    return [compute_absorption_auto, compute_absorption_manual]