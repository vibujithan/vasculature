import numpy as np
from scipy.ndimage import zoom
from skimage.exposure import rescale_intensity as ri


def to_float(img):
    img = img.astype('f4')
    return img


def rescale_intensity(img, in_range=(0, 100)):
    return ri(img, in_range=in_range, out_range=(0, 1))


def resize_block(img, scale):
    return zoom(img, scale)


def invert(img):
    return 1 - img


def intensity_norm(img, imin, imax, gamma=1):
    return ((img - imin) / (imax - imin)) ** gamma


def intensity_clip(img, imin, imax):
    return np.clip(img, imin, imax)


def threshold_mean(img, threshold):
    img[img < threshold] = np.nan
    return np.atleast_3d(np.nanmean(img, axis=(0, 2)))
