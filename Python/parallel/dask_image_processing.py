import json
import math
import os

import SimpleITK as sitk
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import zarr

from tune.adjustments import adjust
from tune.analysis import analyse
from tune.deconvolution import deconvolve
from tune.filters import filter
from tune.parallel import intensity_adjustments
from tune.parallel import utilities
from tune.segmentation import segment

plt.set_loglevel("info")
plt.rcParams['figure.figsize'] = [15, 5]
plt.rcParams['image.cmap'] = 'gray'


class Image:

    def __init__(self, dask_img, file_path=None):
        self.dask_img = dask_img
        self.shape = dask_img.shape
        self.file_path = file_path

    @classmethod
    def from_zarr(cls, file_path, chunk_size=None):
        zarr_img = zarr.open(zarr.storage.NestedDirectoryStore(file_path), mode='r')

        if chunk_size is None:
            chunk_size = zarr_img.chunks

        return cls(da.from_zarr(zarr_img, chunks=chunk_size), file_path=file_path)

    @classmethod
    def from_tif_block(cls, file_path, chunk_size=None):
        zarr_file_path = utilities.from_single_tif(file_path, chunk_size)
        return cls.from_zarr(zarr_file_path)

    @classmethod
    def from_tif_slices(cls, file_path, chunk_size):
        pass  # TODO

    def rechunk(self, chunk_size):
        self.dask_img = self.dask_img.rechunk(chunk_size)

    def sample(self, img_range, prefix='sample', chunk_size=None):
        slice_img_range = tuple([slice(*i) for i in img_range])

        if chunk_size is None:
            chunk_size = self.dask_img.chunksize

        save_path = os.path.join(os.path.dirname(self.file_path), prefix,
                                 prefix + '_' + os.path.basename(self.file_path))

        if not os.path.exists(os.path.join(os.path.dirname(self.file_path), prefix)):
            os.makedirs(os.path.join(os.path.dirname(self.file_path), prefix))

        shape = tuple(map(lambda x, c: (math.ceil(x / c) * c), self.dask_img[slice_img_range].shape, chunk_size))
        store = zarr.NestedDirectoryStore(save_path)
        z_out = zarr.create(shape, chunks=chunk_size, dtype=self.dask_img.dtype,
                            store=store,
                            overwrite=True, fill_value=0)

        temp = self.dask_img[slice_img_range]

        da.to_zarr(temp, z_out)

        filename, ext = os.path.splitext(save_path)

        metadata = {
            'range': img_range,
            'chunk_size': chunk_size
        }

        metadata_file = filename + '.json'
        with open(metadata_file, 'w') as outfile:
            json.dump(metadata, outfile)

        print(save_path)
        return save_path

    def overlap(self, overlap):
        self.dask_img = da.overlap.overlap(self.dask_img, depth={0: overlap[0], 1: overlap[1], 2: overlap[2]},
                                           boundary='reflect')

    def crop(self, img_range):
        slice_img_range = tuple([slice(*i) for i in img_range])
        return Image(self.dask_img[slice_img_range])

    def info(self):
        print("image_path:", self.file_path)
        return self.dask_img

    def trim_overlap(self, overlap):
        self.dask_img = da.overlap.trim_internal(self.dask_img, {0: overlap[0], 1: overlap[1], 2: overlap[2]})

    def visualize(self, plane='subplots', slice_no=100, img_start=(0, 0), img_end=(None, None), clim=None):
        if plane in 'XY':
            fig, ax1 = plt.subplots(1, 1)
            im = ax1.imshow(self.dask_img[slice_no, img_start[0]:img_end[0], img_start[1]:img_end[1]], clim=clim)
            plt.colorbar(im)
            ax1.set_title(plane)
            plt.show()

        elif plane in 'YZ':
            fig, ax1 = plt.subplots(1, 1)
            im = plt.imshow(self.dask_img[img_start[0]:img_end[0], slice_no, img_start[1]:img_end[1]], clim=clim)
            plt.colorbar(im)
            ax1.set_title(plane)
            plt.show()

        elif plane in 'XZ':
            fig, ax1 = plt.subplots(1, 1)
            im = plt.imshow(self.dask_img[img_start[0]:img_end[0], img_start[1]:img_end[1], slice_no], clim=clim)
            plt.colorbar(im)
            ax1.set_title(plane)
            plt.show()

        else:
            fig = plt.figure()
            a = fig.add_subplot(1, 3, 1)
            plt.imshow(self.dask_img[:, :, slice_no], clim=clim)
            a.set_title('XZ')

            a = fig.add_subplot(1, 3, 2)
            plt.imshow(self.dask_img[:, slice_no, :], clim=clim)
            a.set_title('YZ')

            a = fig.add_subplot(1, 3, 3)
            plt.imshow(self.dask_img[slice_no, :, :], clim=clim)
            a.set_title('XY')
            plt.show()

    def compare(self, other_img, plane='XY', slice_no=100, img_start=(0, 0), img_end=(None, None), clim=None):
        if plane in 'XY':
            fig, axs = plt.subplots(1, 2)
            im = axs[0].imshow(self.dask_img[slice_no, img_start[0]:img_end[0], img_start[1]:img_end[1]], clim=clim)
            plt.colorbar(im, ax=axs[0])
            axs[0].set_title(plane)

            im2 = axs[1].imshow(other_img.dask_img[slice_no, img_start[0]:img_end[0], img_start[1]:img_end[1]],
                                clim=clim)
            plt.colorbar(im2, ax=axs[1])
            axs[1].set_title(plane + " of the other image")

        elif plane in 'YZ':
            fig, axs = plt.subplots(1, 2)
            im = axs[0].imshow(self.dask_img[img_start[0]:img_end[0], slice_no, img_start[1]:img_end[1]], clim=clim)
            plt.colorbar(im, ax=axs[0])
            axs[0].set_title(plane)

            im2 = axs[1].imshow(other_img.dask_img[img_start[0]:img_end[0], slice_no, img_start[1]:img_end[1]],
                                clim=clim)
            plt.colorbar(im2, ax=axs[1])
            axs[1].set_title(plane + " of the other image")

        elif plane in 'XZ':
            fig, axs = plt.subplots(1, 2)
            im = axs[0].imshow(self.dask_img[img_start[0]:img_end[0], img_start[1]:img_end[1], slice_no], clim=clim)
            plt.colorbar(im, ax=axs[0])
            axs[0].set_title(plane)

            im2 = axs[1].imshow(other_img.dask_img[img_start[0]:img_end[0], img_start[1]:img_end[1], slice_no],
                                clim=clim)
            plt.colorbar(im2, ax=axs[1])
            axs[1].set_title(plane + " of the other image")

    def to_float(self):
        self.dask_img = da.map_blocks(adjust.to_float, self.dask_img, dtype='f4')

    def deconvolve(self, H, Ht, num_iter=20):
        self.dask_img = da.map_blocks(deconvolve.deconvolve, self.dask_img, H, Ht, num_iter, dtype='f4')

    def anisotropic_diffusion(self, niter=3, kappa=50, gamma=0.2, voxelspacing=None):
        self.dask_img = da.map_blocks(filter.anisotropic_diffusion, self.dask_img, niter=niter,
                                      kappa=kappa, gamma=gamma, voxelspacing=voxelspacing, dtype='f4')

    def wavelet_denoising(self):
        self.dask_img = da.map_blocks(filter.wavelet_denoising, self.dask_img, dtype='f4')

    def to_zarr(self, prefix, chunk_size=None, save_path=None, dtype=None):
        if chunk_size is None:
            chunk_size = self.dask_img.chunksize
        if dtype is None:
            dtype = self.dask_img.dtype
        if save_path is None:
            save_path = os.path.join(os.path.dirname(self.file_path), prefix + '_' + os.path.basename(self.file_path))

        shape = tuple(map(lambda x, c: (math.ceil(x / c) * c), self.dask_img.shape, chunk_size))

        store_save = zarr.NestedDirectoryStore(save_path)
        zarr_out = zarr.create(shape, chunks=chunk_size, store=store_save, dtype=dtype, fill_value=0,
                               overwrite=True)

        da.to_zarr(self.dask_img, zarr_out)
        return save_path

    def compute(self):
        return Image(da.from_array(self.dask_img.compute(), chunks=self.dask_img.chunksize))

    def normalize(self, alpha=1, method='norm', gamma=1, beta=0.5):
        chunks = self.dask_img.chunksize

        w = intensity_adjustments.construct_weights(chunks)
        da_mean_overlap, da_std_overlap = intensity_adjustments.construct_fields(self.dask_img)

        self.dask_img = da.map_blocks(intensity_adjustments.normalization, self.dask_img, w, da_mean_overlap,
                                      da_std_overlap, chunks, alpha=alpha, method=method, gamma=gamma, beta=beta,
                                      dtype='f4')

    def norma(self, alpha=1, gamma=1):
        self.dask_img = da.map_blocks(intensity_adjustments.norma, self.dask_img, alpha=alpha, gamma=gamma, dtype='f4')

    def morph_snake(self, iterations=5, init='threshold', init_val=None):
        self.dask_img = da.map_blocks(segment.morph_snake, self.dask_img, iterations, init, init_val,
                                      dtype='|b')

    def otsu_threshold(self):
        self.dask_img = da.map_blocks(segment.otsu_3d, self.dask_img, dtype='|b')

    def to_tiff_seg(self, save_path=None):
        if save_path is None:
            save_path = os.path.splitext(self.file_path)[0] + '.tif'

        itkimage = sitk.Cast(sitk.GetImageFromArray(255 * self.dask_img), sitk.sitkUInt8)
        itkimage.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(itkimage, save_path, True)

    def to_tiff_slices(self, save_path=None):
        if save_path is None:
            save_path = os.path.splitext(self.file_path)[0] + '.tif'
        itkimage = sitk.Cast(sitk.GetImageFromArray(255 * self.dask_img), sitk.sitkUInt8)
        itkimage.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(itkimage, save_path, True)

    def resize(self, scale):
        new_chunk_size = tuple(map(lambda s, c: int(round(s * c)), self.dask_img.chunksize, scale))
        self.dask_img = da.map_blocks(adjust.resize_block, self.dask_img, scale, chunks=new_chunk_size)

    def LoG_filter(self, sigma):
        self.dask_img = da.map_blocks(filter.log_filtering, self.dask_img, sigma)

    def sobel_filter(self):
        self.dask_img = da.map_blocks(filter.sobel_filter, self.dask_img, dtype='f4')

    def invert(self, in_range=(0, 100)):
        # xz_planes = self.dask_img.rechunk((self.dask_img.shape[0], self.dask_img.shape[1], 1))
        self.dask_img = da.map_blocks(adjust.invert, self.dask_img, in_range=in_range,
                                      dtype=self.dask_img.dtype)

    def frangi(self, sigma_range=range(10, 15), alpha=0.1, beta=0.5, c=0.1, gamma=0.5):
        self.dask_img = da.map_blocks(filter.frangi, self.dask_img, sigma_range=sigma_range,
                                      alpha=alpha, beta=beta, c=c, gamma=gamma, dtype='f4')

    def clean_up(self, threshold):
        self.dask_img = da.map_blocks(analyse.clean_up, self.dask_img, threshold, dtype='|b')

    def median_filter(self, size=5):
        self.dask_img = da.map_blocks(filter.median_filtering, self.dask_img, size)

    def boundary_mask(self, threshold=0.5, iterations=20):
        self.dask_img = da.map_blocks(analyse.create_boundary_mask, self.dask_img, threshold, iterations)

    def direction_volume(self, sigma, rho):
        return da.map_blocks(analyse.direction_vector, self.dask_img, sigma=sigma, rho=rho, is_vol=True,
                             chunks=(30, 30, 30))

    def direction_vector(self, sigma, rho):
        return da.map_blocks(analyse.direction_vector, self.dask_img, sigma=sigma, rho=rho, is_vol=False,
                             chunks=(1, 1, 1), dtype='f4')

    def rescale_intensity(self, in_range=(0, 100)):
        self.dask_img = da.map_blocks(adjust.rescale_intensity, self.dask_img, in_range=in_range, dtype='f4')

    def gaussian(self, sigma):
        self.dask_img = da.map_blocks(filter.gaussian, self.dask_img, sigma=sigma, dtype='f4')

    def boundaries(self, in_range=(0, 100)):
        self.dask_img = self.dask_img.rechunk((self.dask_img.shape[0], self.dask_img.shape[1], 1))
        self.dask_img = da.map_blocks(analyse.find_foreground, self.dask_img, in_range=in_range, dtype='|b')

    def surface_points(self, is_top=True):
        return np.squeeze(
            da.map_blocks(analyse.surface_points, self.dask_img, is_top=is_top, dtype='f4').compute())

    def find_foreground(self, in_range=(0, 100), sigma=7, markers_range=(0.6, 0.9), limits=(-50, 500)):
        if limits[0] is None:
            temp = self.dask_img
        elif limits[0] < 0:
            temp = self.dask_img[:limits[1], :, :]
            top = da.from_array(np.zeros((-limits[0], self.dask_img.shape[1], self.dask_img.shape[2])))
            data = [top, temp]
            temp = da.concatenate(data, axis=0)
        elif limits[1] > self.dask_img.shape[0]:
            temp = np.zeros(limits[1] - limits[0], self.dask_img.shape[1], self.dask_img.shape[2])
            temp[:self.dask_img.shape[0] - limits[0], :, :] = self.dask_img[limits[0]:self.dask_img.shape[0], :, :]
        else:
            temp = self.dask_img[limits[0]:limits[1], :, :]

        xz_planes = temp.rechunk((temp.shape[0], self.dask_img.shape[1], 1))
        return Image(da.map_blocks(analyse.find_foreground, xz_planes, in_range=in_range, sigma=sigma,
                                   markers_range=markers_range,
                                   dtype='|b'))

    def estimate_surface(self, fg, is_top=True, sigma=80):
        points = np.squeeze(
            da.map_blocks(analyse.surface_points, fg.dask_img, is_top=is_top, dtype='f4').compute())
        smoothed_points = analyse.smooth_surface_points(points, is_top, sigma)

        return points, smoothed_points

    def flatten(self, surface, is_top=True, x_chunk=256, limits=(150, 300)):
        columns = self.dask_img.rechunk((self.dask_img.shape[0], x_chunk, 1))
        boundaries = da.from_array(surface, chunks=(x_chunk, 1))

        if is_top:
            depth = limits[1] - (round(np.min(surface)) + limits[0])
            self.dask_img = da.map_blocks(analyse.flatten_top, columns, boundaries, depth, limits=limits,
                                          chunks=(depth, x_chunk, 1),
                                          dtype=self.dask_img.dtype)
        else:
            depth = round(np.max(surface)) + limits[0]
            self.dask_img = da.map_blocks(analyse.flatten_bottom, columns, boundaries, depth, limits=limits,
                                          chunks=(depth, x_chunk, 1),
                                          dtype=self.dask_img.dtype)

    def vessel_density(self):
        self.dask_img = da.map_blocks(analyse.vessel_den, self.dask_img, chunks=(1, 1, 1), dtype='f4')

    def intensity_clip(self, lp=0.02, up=0.02):

        min_val = np.min(self.dask_img).compute()
        max_val = np.max(self.dask_img).compute()
        print("Image range:", (min_val, max_val))

        bins = np.maximum(32000, math.ceil(max_val - min_val))
        h, bins = da.histogram(self.dask_img, bins=bins, range=[min_val, max_val])
        hist = h.compute()

        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]

        # class means for all possible thresholds
        mean1 = np.cumsum(hist * bins[:-1]) / weight1
        mean2 = (np.cumsum((hist * bins[:-1])[::-1]) / weight2[::-1])[::-1]

        # Clip ends to align class 1 and class 2 variables:
        # The last value of ``weight1``/``mean1`` should pair with zero values in
        # ``weight2``/``mean2``, which do not exist.
        variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

        idx = np.argmax(variance12)
        threshold = bins[:-1][idx]

        cumsum = np.cumsum(hist)
        cumsum_percent = cumsum / np.max(cumsum)

        if lp > 0:
            imin = bins[next(i for i, v in enumerate(cumsum_percent) if v > lp)]
        else:
            imin = min_val

        if up < 1:
            imax = bins[next(i for i, v in enumerate(cumsum_percent) if v > 1 - up)]
        else:
            imax = max_val

        print("Clipped image range:", (imin, imax))
        print("Otsu threshold:", threshold)

        self.dask_img = da.map_blocks(adjust.intensity_clip, self.dask_img, imin, imax, dtype='f4')
        return imin, imax, threshold

    def find_otsu(self):

        h, bins = da.histogram(self.dask_img, bins=100, range=[0, 1])
        hist = h.compute()

        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]

        # class means for all possible thresholds
        mean1 = np.cumsum(hist * bins[:-1]) / weight1
        mean2 = (np.cumsum((hist * bins[:-1])[::-1]) / weight2[::-1])[::-1]

        # Clip ends to align class 1 and class 2 variables:
        # The last value of ``weight1``/``mean1`` should pair with zero values in
        # ``weight2``/``mean2``, which do not exist.
        variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

        idx = np.argmax(variance12)
        threshold = bins[:-1][idx]

        print("Otsu threshold:", threshold)

        return threshold

    def save_as(self, prefix='cp', chunk_size=None, dtype=None, save_path=None):
        if chunk_size is None:
            chunk_size = self.dask_img.chunksize
        if dtype is None:
            dtype = self.dask_img.dtype
        if save_path is None:
            save_path = os.path.join(os.path.dirname(self.file_path), prefix + '_' + os.path.basename(self.file_path))

        store_save = zarr.NestedDirectoryStore(save_path)
        zarr_out = zarr.create(self.dask_img.shape, chunks=chunk_size, store=store_save, dtype=dtype,
                               overwrite=True)

        da.to_zarr(self.dask_img, zarr_out)
        return save_path

    def intensity_norm(self, imin, imax, gamma=1):
        self.dask_img = da.map_blocks(adjust.intensity_norm, self.dask_img, imin, imax, gamma=gamma,
                                      dtype='f4')

    def region_growing(self, threshold, iteration=1, multiplier=1, radius=1, replace=1):
        self.dask_img = da.map_blocks(segment.region_growing, self.dask_img, threshold, iteration=iteration,
                                      multiplier=multiplier, radius=radius, replace=replace, dtype='|b')

    def measure_line_profile(self, threshold=0.1):
        rechunked = self.dask_img.rechunk((self.dask_img.chunksize[0], 2048, self.dask_img.chunksize[2]))
        mean_img = da.map_blocks(_threshold_mean, rechunked, threshold, chunks=(1, 2048, 1))
        return da.nanmean(mean_img, axis=(0, 2)).compute()

    def adjust_line_profile(self, threshold=0.1):
        mean_line = self.measure_line_profile(threshold=threshold)

        coefficients = np.polyfit(np.arange(0, 2048), mean_line, 3)
        poly = np.poly1d(coefficients)
        mean_line_fit = poly(np.arange(0, 2048))
        max_fit_val = np.max(mean_line_fit)
        adjustment = max_fit_val / mean_line
        self.dask_img = self.dask_img * np.expand_dims(adjustment, axis=(0, 2))


def _threshold_mean(img, threshold):
    img[img < threshold] = np.nan
    return np.atleast_3d(np.nanmean(img, axis=(0, 2)))
