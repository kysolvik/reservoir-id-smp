import numpy as np


def calc_border_ar(box_height, box_width):
    if 2*box_height + 2*box_width > np.iinfo(np.uint16).max:
        border_dtype = np.int32
    else:
        border_dtype = np.int16
    border_ar = np.zeros((box_height, box_width), dtype=border_dtype)

    border_ar[0, :] = np.arange(border_ar.shape[1]) + 1

    border_ar[:, -1] = np.arange(border_ar.shape[0]) + border_ar.shape[1]

    border_ar[-1, :] = -1 * border_ar[0,:]

    border_ar[1:-1, 0] = -1 * border_ar[1:-1, -1]

    return border_ar
