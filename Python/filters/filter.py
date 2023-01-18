import numpy as np
from medpy.filter.smoothing import anisotropic_diffusion as ad
from scipy import ndimage
from scipy.ndimage import white_tophat, gaussian_filter
from scipy.ndimage.filters import gaussian_laplace, sobel
from skimage.restoration import denoise_wavelet, estimate_sigma

from tune.filters.frangi import vesselness


def frangi(img, sigma_range=range(1, 7), alpha=0.5, beta=0.5, c=1, gamma=1):
    img_out = np.zeros_like(img)
    for sigma in sigma_range:
        print(sigma)
        img_out = np.maximum(img_out, vesselness(img, sigma, alpha, beta, c, gamma))
    return img_out


def gaussian(img, sigma):
    return gaussian_filter(img, sigma=sigma)


def anisotropic_diffusion(img, niter=3, kappa=20, gamma=0.1, voxelspacing=None):
    return ad(img, niter=niter, kappa=kappa, gamma=gamma, voxelspacing=voxelspacing)


def wavelet_denoising(img):
    sigma_est = estimate_sigma(img, multichannel=False, average_sigmas=True)
    return denoise_wavelet(img, multichannel=False, method='VisuShrink', mode='soft', sigma=sigma_est / 2,
                           rescale_sigma=True)


def bright_spot_removal(img, scale):
    return img - white_tophat(img, scale)


def log_filtering(img, sigma):
    return gaussian_laplace(img, sigma=sigma)


def sobel_filter(img):
    return sobel(img)


def median_filtering(img, size=20):
    return ndimage.median_filter(img, size)
