import numpy as np


def Linear_Warp(img, warp_list, weight_list, padding_value=0):

    warp2_i0, warp2_j0, warp2_i1, warp2_j1 = warp_list
    weight2_i0, weight2_i1, weight2_j0, weight2_j1 = weight_list

    height, width = img.shape[0], img.shape[1]

    img = np.pad(img, ((1, 1), (1, 1), (0, 0)), constant_values=padding_value)

    res2_i0j0 = img[warp2_i0, warp2_j0]
    res2_i1j0 = img[warp2_i1, warp2_j0]
    res2_i0j1 = img[warp2_i0, warp2_j1]
    res2_i1j1 = img[warp2_i1, warp2_j1]

    res2_j0 = res2_i0j0 * weight2_i0 + res2_i1j0 * weight2_i1
    res2_j1 = res2_i0j1 * weight2_i0 + res2_i1j1 * weight2_i1
    res2 = res2_j0 * weight2_j0 + res2_j1 * weight2_j1
    res2 = res2.reshape((height, width, img.shape[2])).astype(img.dtype)

    return res2


def cal_warp_index_weight(warp_index, max_height, max_width):
    warp_i, warp_j = warp_index

    warp_i0 = np.floor(warp_i)
    warp_i1 = np.floor(warp_i) + 1
    warp_j0 = np.floor(warp_j)
    warp_j1 = np.floor(warp_j) + 1

    weight_i0 = (1 - np.abs(warp_i0 - warp_i))[:, np.newaxis]
    weight_i1 = (1 - np.abs(warp_i1 - warp_i))[:, np.newaxis]
    weight_j0 = (1 - np.abs(warp_j0 - warp_j))[:, np.newaxis]
    weight_j1 = (1 - np.abs(warp_j1 - warp_j))[:, np.newaxis]

    # clamp out of screen indexes
    warp_i0 = np.clip(warp_i0, 0, max_height + 1).astype(np.long)
    warp_i1 = np.clip(warp_i1, 0, max_height + 1).astype(np.long)
    warp_j0 = np.clip(warp_j0, 0, max_width  + 1).astype(np.long)
    warp_j1 = np.clip(warp_j1, 0, max_width  + 1).astype(np.long)

    return [weight_i0, weight_i1, weight_j0, weight_j1], [warp_i0, warp_i1, warp_j0, warp_j1]


def warp_img(img, motion_vector, padding_value=0):

    height, width = img.shape[0], img.shape[1]

    flat_index = np.arange(height*width)
    i = flat_index // width
    j = flat_index - i * width

    # Because we will pad original image in warp
    i += 1
    j += 1

    # motion vector record the change in SCREEN SPACE! and indexed by XY!
    # ↑
    # ↑
    # ↑
    # ↑
    # ↑
    # ↑
    # ↑ → → → → → → → →
    # when we index the array by i, j the space is like that
    # ↓ → → → → → → → →
    # ↓
    # ↓
    # ↓
    # ↓
    # ↓
    # ↓
    # keep in mind with that

    warp_i = i + motion_vector[i, j, 1]
    warp_j = j - motion_vector[i, j, 0]

    weight_list, warp_list = cal_warp_index_weight([warp_i, warp_j], height, width)

    warpped_img = Linear_Warp(img, warp_list, weight_list, padding_value)

    return warpped_img




