import glob
import json
import math
import os
import time

import SimpleITK as sitk
import dask
import dask.array as da
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.io import savemat
from scipy.ndimage import uniform_filter1d
from skimage import io

from tune.adjustments.adjust import intensity_norm
from tune.analysis.analyse import direction_vector
from tune.analysis.direction_analysis import make_volume
from tune.deconvolution.psf import PSF


def deconvolve(img, img_resolution, chunk_size, prefix, overlap, psf_invert, psf_path, psf_res, iterations,
               save_chunk_size, client, save_path=None):
    start_time = time.time()

    psf = PSF(psf_path, psf_res)
    if psf_invert:
        psf.flip()
    psf.visualize("PSF extracted")

    psf.adjust_resolution(img_resolution)
    psf.resize(chunk_size, overlap)

    H, Ht = psf.calculate_otf()
    print("OTF calculated")

    img.to_float()
    img.overlap(overlap)
    img.deconvolve(H, Ht, num_iter=iterations)
    img.trim_overlap(overlap)

    deconv_img_path = img.to_zarr(prefix=prefix, chunk_size=save_chunk_size, save_path=save_path)
    elapsed_time = time.time() - start_time
    print("Elapsed time:", elapsed_time)

    parent_dir, name = os.path.split(img.file_path)
    filename, ext = os.path.splitext(deconv_img_path)

    metadata = {
        'img_in': {
            'name': name,
            'res': img_resolution,
        },

        'process': {
            'name': 'deconv',
            'chunk_size': save_chunk_size,
            'overlap': overlap,
            'psf_res': psf_res,
            'psf_path': psf_path,
            'iterations': iterations
        },

        'img_out': {
            'name': os.path.basename(deconv_img_path),
            'res': img_resolution,
            'elapsed_time': round(elapsed_time),
            'hostname': os.uname()[1],
            'workers': len(client.ncores()),
            'threads': next(iter(client.nthreads().values()))
        }
    }

    metadata_file = filename + '.json'
    with open(metadata_file, 'w') as outfile:
        json.dump(metadata, outfile)

    print(deconv_img_path)
    return deconv_img_path


def bg_correction(img, bg_percentile, vertical_window, mean_chunk_size, bg_chunks, client, prefix='bg'):
    start_time = time.time()

    img.intensity_clip(lp=bg_percentile, up=bg_percentile)

    mean_img = img.dask_img.mean(axis=[2])
    mean_img = uniform_filter1d(mean_img, size=vertical_window, axis=0)

    mean_line = da.from_array(np.atleast_3d(mean_img), chunks=mean_chunk_size)

    plt.imshow(mean_line)
    plt.colorbar()

    bg_img = img.dask_img - mean_line

    shape = tuple(map(lambda x, c: (math.ceil(x / c) * c), bg_img.shape, bg_chunks))

    bg_save_path = os.path.join(os.path.dirname(img.file_path), prefix + '_' + os.path.basename(img.file_path))

    store = zarr.NestedDirectoryStore(bg_save_path)
    z_out = zarr.create(shape, chunks=bg_chunks, dtype=bg_img.dtype,
                        store=store,
                        overwrite=True, fill_value=0)

    da.to_zarr(bg_img, z_out)

    elapsed_time = time.time() - start_time
    print("Elapsed time:", elapsed_time)

    parent_dir, name = os.path.split(img.file_path)
    filename, ext = os.path.splitext(bg_save_path)

    metadata = {
        'img_in': {
            'name': name,
        },

        'process': {
            'name': 'bg',
            'bg_percentile': bg_percentile,
            'vertical_window': vertical_window,
            'bg_chunks': bg_chunks,
        },

        'img_out': {
            'name': os.path.basename(bg_save_path),
            'elapsed_time': round(elapsed_time),
            'hostname': os.uname()[1],
            'workers': len(client.ncores()),
            'threads': next(iter(client.nthreads().values()))
        }
    }

    metadata_file = filename + '.json'
    with open(metadata_file, 'w') as outfile:
        json.dump(metadata, outfile)

    print(bg_save_path)
    return bg_save_path


def diffuse(img, overlap, diffuse_niter, diffuse_kappa, diffuse_gamma, diffuse_voxels, save_chunk_size, client,
            prefix='diff'):
    start_time = time.time()

    img.overlap(overlap)
    img.anisotropic_diffusion(niter=diffuse_niter, kappa=diffuse_kappa, gamma=diffuse_gamma,
                              voxelspacing=diffuse_voxels)
    img.trim_overlap(overlap)

    diff_img_path = img.to_zarr(prefix=prefix, chunk_size=save_chunk_size)
    elapsed_time = time.time() - start_time
    print("Elapsed time:", elapsed_time)

    parent_dir, name = os.path.split(img.file_path)
    filename, ext = os.path.splitext(diff_img_path)

    metadata = {
        'img_in': {
            'name': name,
        },

        'process': {
            'name': 'diffuse',
            'chunk_size': save_chunk_size,
            'overlap': overlap,
            'niter': diffuse_niter,
            'kappa': diffuse_kappa,
            'gamma': diffuse_gamma,
            'voxelspacing': diffuse_voxels
        },

        'img_out': {
            'name': os.path.basename(diff_img_path),
            'elapsed_time': round(elapsed_time),
            'hostname': os.uname()[1],
            'workers': len(client.ncores()),
            'threads': next(iter(client.nthreads().values()))
        }
    }

    metadata_file = filename + '.json'
    with open(metadata_file, 'w') as outfile:
        json.dump(metadata, outfile)

    print(diff_img_path)
    return diff_img_path


def gauss(img, overlap, sigma, save_chunk_size, client, prefix='gauss'):
    start_time = time.time()

    img.overlap(overlap)
    img.gaussian(sigma)
    img.trim_overlap(overlap)

    diff_img_path = img.to_zarr(prefix=prefix, chunk_size=save_chunk_size)
    elapsed_time = time.time() - start_time
    print("Elapsed time:", elapsed_time)

    parent_dir, name = os.path.split(img.file_path)
    filename, ext = os.path.splitext(diff_img_path)

    metadata = {
        'img_in': {
            'name': name,
        },

        'process': {
            'name': 'diffuse',
            'chunk_size': save_chunk_size,
            'overlap': overlap,
            'sigma': sigma
        },

        'img_out': {
            'name': os.path.basename(diff_img_path),
            'elapsed_time': round(elapsed_time),
            'hostname': os.uname()[1],
            'workers': len(client.ncores()),
            'threads': next(iter(client.nthreads().values()))
        }
    }

    metadata_file = filename + '.json'
    with open(metadata_file, 'w') as outfile:
        json.dump(metadata, outfile)

    print(diff_img_path)
    return diff_img_path


def normalize(img, rescale_gamma, rescale_lp, rescale_up, save_chunk_size, client, prefix='norm'):
    start_time = time.time()

    imin, imax, threshold = img.intensity_clip(lp=rescale_lp, up=rescale_up)
    img.intensity_norm(imin, imax, gamma=rescale_gamma)
    threshold = intensity_norm(threshold, imin, imax, gamma=rescale_gamma)

    norm_img_path = img.to_zarr(prefix=prefix, chunk_size=save_chunk_size)
    elapsed_time = time.time() - start_time
    print("Elapsed time:", elapsed_time)

    parent_dir, name = os.path.split(img.file_path)
    filename, ext = os.path.splitext(norm_img_path)

    metadata = {
        'img_in': {
            'name': name,
        },

        'process': {
            'name': 'norm',
            'gamma': rescale_gamma,
            'rescale_lp': rescale_lp,
            'rescale_up': rescale_up
        },

        'img_out': {
            'name': os.path.basename(norm_img_path),
            'elapsed_time': round(elapsed_time),
            'hostname': os.uname()[1],
            'workers': len(client.ncores()),
            'threads': next(iter(client.nthreads().values()))
        }
    }

    metadata_file = filename + '.json'
    with open(metadata_file, 'w') as outfile:
        json.dump(metadata, outfile)

    print(norm_img_path)
    print("Otsu threshold after normalization:", threshold)

    return norm_img_path


def segment(img, overlap, init_val, active_contour, gauss_sigma, cl_threshold, save_chunk_size, client, prefix='seg'):
    start_time = time.time()

    img.overlap(overlap)
    img.gaussian(sigma=gauss_sigma)

    if active_contour > 0:
        img.morph_snake(iterations=active_contour, init='threshold', init_val=init_val)
    else:
        img.dask_img = img.dask_img > init_val

    img.clean_up(threshold=cl_threshold)
    img.trim_overlap(overlap)

    seg_img_path = img.to_zarr(prefix=prefix, chunk_size=save_chunk_size)
    elapsed_time = time.time() - start_time
    print("Elapsed time:", elapsed_time)

    parent_dir, name = os.path.split(img.file_path)
    filename, ext = os.path.splitext(seg_img_path)

    metadata = {
        'img_in': {
            'name': name,
        },

        'process': {
            'name': prefix,
            'overlap': overlap,
            'threshold': init_val,
            'cl_threshold': cl_threshold
        },
        'img_out': {
            'name': os.path.basename(seg_img_path),
            'elapsed_time': round(elapsed_time),
            'hostname': os.uname()[1],
            'workers': len(client.ncores()),
            'threads': next(iter(client.nthreads().values()))
        }
    }

    metadata_file = filename + '.json'
    with open(metadata_file, 'w') as outfile:
        json.dump(metadata, outfile)

    print(seg_img_path)
    return seg_img_path


def region_growing(img, overlap, init_val, grow_iter, gauss_sigma, cl_threshold, save_chunk_size, client,
                   prefix='seg'):
    start_time = time.time()

    img.overlap(overlap)
    img.gaussian(sigma=gauss_sigma)

    if grow_iter > 0:
        img.region_growing(threshold=init_val, iteration=grow_iter)
    else:
        img.dask_img = img.dask_img > init_val

    img.clean_up(threshold=cl_threshold)
    img.trim_overlap(overlap)

    seg_img_path = img.to_zarr(prefix=prefix, chunk_size=save_chunk_size)
    elapsed_time = time.time() - start_time
    print("Elapsed time:", elapsed_time)

    parent_dir, name = os.path.split(img.file_path)
    filename, ext = os.path.splitext(seg_img_path)

    metadata = {
        'img_in': {
            'name': name,
        },

        'process': {
            'name': prefix,
            'overlap': overlap,
            'threshold': init_val,
            'iterations': grow_iter,
            'cl_threshold': cl_threshold
        },
        'img_out': {
            'name': os.path.basename(seg_img_path),
            'elapsed_time': round(elapsed_time),
            'hostname': os.uname()[1],
            'workers': len(client.ncores()),
            'threads': next(iter(client.nthreads().values()))
        }
    }

    metadata_file = filename + '.json'
    with open(metadata_file, 'w') as outfile:
        json.dump(metadata, outfile)

    print(seg_img_path)
    return seg_img_path


def downscale(img, sigma, resize_scale, chunk_size, overlap, client, prefix='ds'):
    start_time = time.time()

    img.to_float()
    img.overlap(overlap)
    img.gaussian(sigma=sigma)
    img.resize(scale=resize_scale)

    ds_overlap = tuple(map(lambda s, c: int(round(s * c)), overlap, resize_scale))
    img.trim_overlap(ds_overlap)

    ds_img_path = img.to_zarr(prefix=prefix, chunk_size=chunk_size)
    elapsed_time = time.time() - start_time
    print("Elapsed time:", elapsed_time)

    filename, ext = os.path.splitext(ds_img_path)
    metadata = {
        'process': {
            'name': prefix,
            'sigma_1': sigma,
            'resize_scale': resize_scale,
            'chunk_size': chunk_size
        },
        'img_out': {
            'name': os.path.basename(ds_img_path),
            'elapsed_time': round(elapsed_time),
            'hostname': os.uname()[1],
            'workers': len(client.ncores()),
            'threads': next(iter(client.nthreads().values()))
        }
    }

    metadata_file = filename + '.json'
    with open(metadata_file, 'w') as outfile:
        json.dump(metadata, outfile)

    print(ds_img_path)
    return ds_img_path


def direction(img, sigma, rho, grad_threshold=1e-3, block=20, interp=1, prefix='dir', save_vol=False):
    start_time = time.time()

    vecs = da.map_blocks(direction_vector, img.dask_img, sigma=sigma, rho=rho, grad_threshold=grad_threshold,
                         chunks=(3, 1, 1), dtype='f4')

    v = vecs.compute()

    save_path = os.path.join(os.path.dirname(img.file_path), prefix + '_' + os.path.basename(img.file_path))

    np.save(save_path, v)
    f = {'myo': v}
    savemat(save_path + '.mat', f)

    if save_vol:
        vol = make_volume(v, block=block, interp=interp)
        vol_save_path = os.path.join(os.path.dirname(img.file_path), prefix + '_' + os.path.basename(img.file_path))

        store_save = zarr.NestedDirectoryStore(vol_save_path)
        zarr_out = zarr.create(vol.shape, chunks=(block, block, block), store=store_save, dtype='|b',
                               overwrite=True)

        zarr_out[:] = vol

    elapsed_time = time.time() - start_time
    print("Elapsed time:", elapsed_time)

    k = np.abs(v)

    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color='white')

    plt.figure(figsize=(6, 6), dpi=300)
    plt.imshow(k[2, :, :], clim=(0, 1), cmap='viridis')

    plt.savefig(save_path + '.png', dpi=300)

    filename, ext = os.path.splitext(save_path)
    metadata = {
        'process': {
            'name': 'vec',
            'sigma': sigma,
            'interp': interp,
            'rho': rho,
            'grad_threshold': grad_threshold,
            'block': block
        }
    }

    metadata_file = filename + '.json'
    with open(metadata_file, 'w') as outfile:
        json.dump(metadata, outfile)

    print(save_path)
    return save_path


def save_tif_slices(filepath):
    start_time = time.time()

    store = zarr.NestedDirectoryStore(filepath)
    z = zarr.open(store, mode='r')

    chunks = (1, z.shape[1], z.shape[2])
    da_input = da.from_zarr(z, chunks=chunks, dtype='bool')

    index = np.arange(0, z.shape[0])
    index = np.expand_dims(index, [1, 2])
    da_index = da.from_array(index, chunks=(1, 1, 1))

    filename, ext = os.path.splitext(filepath)

    if not os.path.exists(filename):
        os.makedirs(filename)

    def save_slice(img, index):
        itkimage = sitk.Cast(sitk.GetImageFromArray(255 * img), sitk.sitkUInt8)
        itkimage.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(itkimage, os.path.join(filename, 'slice_' + str(np.squeeze(index)).zfill(3) + '.tif'), True)
        return np.atleast_3d(1)

    da_out = da.map_blocks(save_slice, da_input, da_index, chunks=(1, 1, 1), dtype='uint8')

    da_out.compute()
    elapsed_time = time.time() - start_time
    print("Elapsed time:", elapsed_time)
    return 1


def save_tif_slices_uint8(filepath):
    start_time = time.time()

    store = zarr.NestedDirectoryStore(filepath)
    z = zarr.open(store, mode='r')

    chunks = (1, z.shape[1], z.shape[2])
    da_input = da.from_zarr(z, chunks=chunks)

    index = np.arange(0, z.shape[0])
    index = np.expand_dims(index, [1, 2])
    da_index = da.from_array(index, chunks=(1, 1, 1))

    filename, ext = os.path.splitext(filepath)

    if not os.path.exists(filename):
        os.makedirs(filename)

    def save_slice(img, index):
        itkimage = sitk.Cast(sitk.GetImageFromArray(255 * img), sitk.sitkUInt8)
        itkimage.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(itkimage, os.path.join(filename, 'slice_' + str(np.squeeze(index)).zfill(3) + '.tif'), True)
        return np.atleast_3d(1)

    da_out = da.map_blocks(save_slice, da_input, da_index, chunks=(1, 1, 1), dtype='uint8')

    da_out.compute()
    elapsed_time = time.time() - start_time
    print("Elapsed time:", elapsed_time)
    return 1


def tif2zarr(json_file_path):
    start_time = time.time()

    with open(json_file_path) as json_file:
        metadata = json.load(json_file)

    path = metadata['path']
    chunks = metadata['chunks']
    z_range = metadata['z_range']
    xy_range = metadata['xy_range']
    dtype = metadata['dtype']

    filenames = sorted(glob.glob(os.path.join(path, "*.tif")))

    z_range = slice(z_range[0], z_range[1])
    xy_range = (slice(xy_range[0][0], xy_range[0][1]), slice(xy_range[1][0], xy_range[1][1]))

    sample = io.imread(filenames[0])

    lazy_arrays = [dask.delayed(io.imread)(fn) for fn in filenames][z_range]
    lazy_arrays = [da.from_delayed(x[xy_range], shape=sample[xy_range].shape, dtype=dtype)
                   for x in lazy_arrays]

    da_img = da.stack(lazy_arrays)

    print('Dask shape:', da_img.shape)

    shape = tuple(map(lambda x, c: (math.ceil(x / c) * c), da_img.shape, chunks))

    store = zarr.NestedDirectoryStore(path + ".zarr")
    z_out = zarr.create(shape=shape, chunks=chunks, dtype=da_img.dtype, store=store, overwrite=True, fill_value=0)

    da.to_zarr(da_img, z_out)

    elapsed_time = time.time() - start_time
    print("Elapsed time:", elapsed_time)
    return z_out.info


def downscale_3(img, sigma, resize_scale, chunk_size, overlap, client, prefix='ds3'):
    start_time = time.time()

    img.to_float()
    img.overlap(overlap)
    img.gaussian(sigma=sigma)
    img.resize(scale=[1, 0.5, 0.5])

    img.gaussian(sigma=sigma)
    img.resize(scale=[0.5, 0.5, 0.5])

    img.gaussian(sigma=sigma)
    img.resize(scale=[0.5, 0.5, 0.5])

    ds_overlap = tuple(map(lambda s, c: int(round(s * c)), overlap, resize_scale))
    img.trim_overlap(ds_overlap)

    ds_img_path = img.to_zarr(prefix=prefix, chunk_size=chunk_size)
    elapsed_time = time.time() - start_time
    print("Elapsed time:", elapsed_time)

    filename, ext = os.path.splitext(ds_img_path)
    metadata = {
        'process': {
            'name': prefix,
            'sigma_1': sigma,
            'resize_scale': resize_scale,
            'chunk_size': chunk_size
        },
        'img_out': {
            'name': os.path.basename(ds_img_path),
            'elapsed_time': round(elapsed_time),
            'hostname': os.uname()[1],
            'workers': len(client.ncores()),
            'threads': next(iter(client.nthreads().values()))
        }
    }

    metadata_file = filename + '.json'
    with open(metadata_file, 'w') as outfile:
        json.dump(metadata, outfile)

    print(ds_img_path)
    return ds_img_path


def save_as_tif(filepath):
    start_time = time.time()
    filename, ext = os.path.splitext(filepath)

    store = zarr.NestedDirectoryStore(filepath)
    z = zarr.open(store, mode='r')
    img = z[:, :, :]

    itkimage = sitk.Cast(sitk.GetImageFromArray(255 * img[:, :, :]), sitk.sitkUInt8)
    itkimage.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(itkimage, filename + '.tif', True)

    elapsed_time = time.time() - start_time
    print("Elapsed time:", elapsed_time)
    return 1
