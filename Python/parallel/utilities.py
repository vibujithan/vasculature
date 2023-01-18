import math
import os

import dask
import dask.array as da
import zarr
from PIL import Image
from skimage import io


def from_single_tif(file_path, chunk_size=None):
    filename, ext = os.path.splitext(file_path)
    save_path = filename + ".zarr"

    dataset = Image.open(file_path)
    sample = io.imread(file_path, img_num=1)

    z_range = dataset.n_frames
    xy_range = (slice(None, None), slice(None, None))

    lazy_arrays = [dask.delayed(io.imread)(file_path, img_num=i) for i in range(z_range)]
    lazy_arrays = [da.from_delayed(x[xy_range], shape=sample[xy_range].shape, dtype=sample.dtype)
                   for x in lazy_arrays]

    da_img = da.stack(lazy_arrays)

    if chunk_size is None:
        chunk_size = (z_range,) + (256, 256)

    shape = tuple(map(lambda x, c: (math.ceil(x / c) * c), da_img.shape, chunk_size))

    store = zarr.NestedDirectoryStore(save_path)
    z_out = zarr.create(shape=shape, chunks=chunk_size, dtype=da_img.dtype, store=store, overwrite=True, fill_value=0)
    da.to_zarr(da_img, z_out)
    return save_path


def from_tif_slices(file_path, chunk_size=None):
    pass
