import SimpleITK as sitk
import morphsnakes as ms
import numpy as np
from skimage.exposure import histogram


def morph_snake(img, iterations=5, init='threshold', init_val=None):
    if init in 'otsu':
        init_ls = otsu_3d(img)
    else:
        init_ls = np.multiply(img > init_val, 1)
    return ms.morphological_chan_vese(img, iterations=iterations, init_level_set=init_ls, smoothing=1, lambda1=1.2,
                                      lambda2=1)


def otsu_3d(image, nbins=256):
    if np.min(image) == np.max(image):
        raise ValueError("threshold_otsu is expected to work with images "
                         "having more than one color. The input image seems "
                         "to have just one color {0}.".format(np.min(image)))

    hist, bin_centers = histogram(image.ravel(), nbins)
    hist = hist.astype(float)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return np.multiply(image > threshold, 1)


def region_growing(img, threshold, iteration=1, multiplier=1, radius=1, replace=1):
    I = sitk.GetImageFromArray(img)
    a = np.argwhere(img > threshold)
    initial_seed_point_indexes = [tuple(map(lambda x: int(round(x)), [x, y, z])) for [z, y, x] in
                                  zip(a[:, 0], a[:, 1], a[:, 2])]

    seg_implicit_thresholds = sitk.ConfidenceConnected(I, seedList=initial_seed_point_indexes,
                                                       numberOfIterations=iteration,
                                                       multiplier=multiplier,
                                                       initialNeighborhoodRadius=radius,
                                                       replaceValue=replace)

    # vector_radius = (1, 1, 1)
    # kernel = sitk.sitkBall
    # cleaned = sitk.BinaryMorphologicalClosing(seg_implicit_thresholds, vector_radius, kernel)

    return sitk.GetArrayFromImage(seg_implicit_thresholds)
