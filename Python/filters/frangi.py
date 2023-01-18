import numpy as np
from scipy.ndimage.filters import gaussian_filter


def hessian_volume(img, sigma):
    if sigma != 0:
        img = gaussian_filter(img, sigma=sigma)

    Gx = np.gradient(img, axis=0)
    Gy = np.gradient(img, axis=1)
    Gz = np.gradient(img, axis=2)

    Gxx = np.gradient(Gx, axis=0)
    Gxy = np.gradient(Gx, axis=1)
    Gxz = np.gradient(Gx, axis=2)

    Gyy = np.gradient(Gy, axis=1)
    Gyz = np.gradient(Gy, axis=1)

    Gzz = np.gradient(Gz, axis=2)

    return Gxx, Gyy, Gzz, Gxy, Gxz, Gyz


def eigen_values(a11, a12, a13, a22, a23, a33):
    ep = 1e-50

    b = a11 + ep
    d = a22 + ep
    j = a33 + ep

    c = -(a12 ** 2 + a13 ** 2 + a23 ** 2 - b * d - d * j - j * b)
    d = -(b * d * j - (a23 ** 2) * b - (a12 ** 2) * j - (a13 ** 2) * d + 2 * a13 * a12 * a23)

    b = -a11 - a22 - a33 - ep
    d = d + ((2 * b ** 3) - (9 * b * c)) / 27

    c = (b ** 2) / 3 - c
    c = c ** 3
    c = c / 27
    c = np.maximum(c, 0)
    c = np.sqrt(c)

    j = c ** (1 / 3)
    c = c + (c == 0)
    d = -d / 2 / c
    d = np.minimum(d, 1)
    d = np.maximum(d, -1)

    d = np.real(np.arccos(d) / 3)
    c = j * np.cos(d)
    d = j * np.sqrt(3) * np.sin(d)
    b = -b / 3

    j = -c - d + b
    d = -c + d + b
    b = 2 * c + b

    return np.stack((j, d, b))


def vesselness(img, sigma, alpha, beta, c, gamma):
    (Gxx, Gyy, Gzz, Gxy, Gxz, Gyz) = hessian_volume(img, sigma=sigma)

    if sigma > 0:
        scale = sigma ** gamma
        Gxx *= scale
        Gxy *= scale
        Gxz *= scale
        Gyy *= scale
        Gyz *= scale
        Gzz *= scale

    lambdas = eigen_values(Gxx, Gxy, Gxz, Gyy, Gyz, Gzz)
    lambdas_abs = np.absolute(lambdas)

    abs_ind = np.argsort(lambdas_abs, axis=0)
    lambdas_abs = np.take_along_axis(lambdas_abs, abs_ind, axis=0)
    lambdas = np.take_along_axis(lambdas, abs_ind, axis=0)

    lambda_abs1 = lambdas_abs[0, :, :, :]
    lambda_abs2 = lambdas_abs[1, :, :, :]
    lambda_abs3 = lambdas_abs[2, :, :, :]

    ra = lambda_abs2 / (lambda_abs3 + 1e-10)
    rb = lambda_abs1 / (np.sqrt(lambda_abs2 * lambda_abs3) + 1e-10)
    S = np.sqrt(lambda_abs1 ** 2 + lambda_abs2 ** 2 + lambda_abs3 ** 2)

    A = 2 * alpha ** 2
    B = 2 * beta ** 2
    C = 2 * c ** 2

    exp_ra = (1 - np.exp(-(ra ** 2 / A)))
    exp_rb = np.exp(-(rb ** 2 / B))
    exp_s = (1 - np.exp(-S ** 2 / C))

    voxel_data = exp_ra * exp_rb * exp_s

    voxel_data[lambdas[1, :, :, :] > 0] = 0
    voxel_data[lambdas[2, :, :, :] > 0] = 0

    return voxel_data
