import dask.array as da
import numpy as np
from scipy import spatial


def mean_block(img):
    return np.atleast_3d(np.mean(img))


def std_block(img):
    return np.atleast_3d(np.std(img))


def construct_fields(da_input):
    da_mean = da.map_blocks(mean_block, da_input, chunks=(1, 1, 1), dtype='f')
    da_std = da.map_blocks(std_block, da_input, chunks=(1, 1, 1), dtype='f')

    da_mean_overlap = da.overlap.overlap(da_mean, depth={0: 1, 1: 1, 2: 1}, boundary='reflect')
    da_std_overlap = da.overlap.overlap(da_std, depth={0: 1, 1: 1, 2: 1}, boundary='reflect')

    return da_mean_overlap, da_std_overlap


def construct_weights(chunks):
    p = np.linspace(chunks[0], 5 * chunks[0], 3)
    r = np.linspace(chunks[1], 5 * chunks[1], 3)
    c = np.linspace(chunks[2], 5 * chunks[2], 3)

    pv, rv, cv = np.meshgrid(p, r, c, indexing='ij')

    mean_coordinates = np.array([pv.ravel(), rv.ravel(), cv.ravel()], dtype='f4').T

    pp = np.arange(2 * chunks[0], 4 * chunks[0], 2)
    pr = np.arange(2 * chunks[1], 4 * chunks[1], 2)
    pc = np.arange(2 * chunks[2], 4 * chunks[2], 2)

    ppv, prv, pcv = np.meshgrid(pp, pr, pc, indexing='ij')
    img_cords = np.array([ppv.ravel(), prv.ravel(), pcv.ravel()], dtype='f4').T

    distance = spatial.distance.cdist(img_cords, mean_coordinates)
    distance = 3 * np.max(chunks) - distance
    distance[distance <= 0] = 0
    w = distance ** 2
    w /= np.sum(w, axis=1)[:, np.newaxis]
    return w


def normalization(img, w, mean_val, std_val, chunks, alpha=1, method='norm', gamma=1, beta=0.5):
    mean_field = np.matmul(w, mean_val.ravel()).reshape(chunks)
    std_field = np.matmul(w, std_val.ravel()).reshape(chunks)

    print('just')

    if method in 'norm':
        return ((img - mean_field) / (alpha * std_field)) ** gamma
    elif method in 'just':
        return img - beta * mean_field
    elif method in 'mean':
        return (1 / (1 + np.exp(- alpha * (img - beta * mean_field)))) ** gamma
    else:
        return (1 / (1 + np.exp(- alpha * (img - beta * mean_field) / std_field))) ** gamma


def norma(img, alpha=1, gamma=1):
    return (1 / (1 + np.exp(- alpha * img))) ** gamma
