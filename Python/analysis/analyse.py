import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import sobel
from scipy.ndimage.morphology import binary_closing
from skimage.exposure import rescale_intensity as ri
from skimage.measure import label, regionprops
from skimage.segmentation import watershed

from tune.analysis.direction_analysis import structure_tensor, mean_axis


def direction_vector(volume, sigma, rho, grad_threshold=1e-3, direction=0):
    S, grad = structure_tensor(volume, sigma, rho)
    mask = grad > grad_threshold
    prom_S = S[mask, ...]
    w, v = np.linalg.eigh(prom_S, UPLO='U')

    vec = np.nan_to_num(v[..., direction])
    vec = mean_axis(vec)

    x, y, z = vec / np.linalg.norm(vec)
    return np.expand_dims([x, y, z], axis=[1, 2])


def create_boundary_mask(img, threshold=0.5, iterations=30):
    return binary_closing(img > threshold, iterations=iterations)


def find_foreground(img, in_range=(0, 100), sigma=7, markers_range=(.6, .9)):
    img = ri(img, in_range=in_range)
    img = gaussian_filter(img, sigma=sigma)

    filled = np.squeeze(img)
    edges = sobel(filled, axis=0)

    markers = np.zeros_like(filled)

    foreground, background = 1, 2
    markers[filled < markers_range[0]] = background
    markers[filled > markers_range[1]] = foreground

    ws = watershed(edges, markers) == foreground

    return np.atleast_3d(ws)


def surface_points(input_img, is_top=True):
    input_img = np.multiply(sobel(input_img, axis=0) > 0.5, 1)
    if is_top:
        max_val = np.max(input_img, axis=0)
        points = np.argmax(input_img, axis=0)
        points[max_val == 0] = -1
    else:
        max_val = np.max(input_img, axis=0)
        points = input_img.shape[0] - np.argmax(np.flip(input_img, axis=0), axis=0)
        points[max_val == 0] = -1

    return np.atleast_3d(points)


def smooth_surface_points(points, is_top=True, sigma=80):
    if is_top:
        upper_bound = np.mean(points[points != -1]) + 1 * np.std(points[points != -1])
        points[points == -1] = upper_bound
        points[points > upper_bound] = upper_bound
        points_hat = gaussian_filter(points, sigma=sigma)
    else:
        lower_bound = np.mean(points[points != -1]) - 1 * np.std(points[points != -1])
        points[points == -1] = lower_bound
        points[points < lower_bound] = lower_bound
        points_hat = gaussian_filter(points, sigma=sigma)

    return points_hat


def flatten_top(img, point, depth, limits):
    flat_vec = np.zeros((depth, img.shape[1], 1))

    for i in range(img.shape[1]):
        vec = img[round(point[i, 0]) + limits[0]:limits[1], i, 0]
        flat_vec[0:vec.shape[0], i, 0] = vec
    return flat_vec


def flatten_bottom(img, point, depth, limits):
    flat_vec = np.zeros((depth, img.shape[1], 1))

    for i in range(img.shape[1]):
        vec = img[:round(point[i, 0]) + limits[0], i, 0]
        flat_vec[-vec.shape[0]:, i, 0] = vec
    return flat_vec


def vessel_den(img):
    print('new')
    return np.atleast_3d(np.sum(img))


def clean_up(img, threshold):
    label_img = label(img, connectivity=3)
    rprop = regionprops(label_img)

    for r in rprop:
        if threshold > r.area > 0:
            img[r.coords.T[0, :], r.coords.T[1, :], r.coords.T[2, :]] = 0
    return img
