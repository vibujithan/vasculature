import matplotlib.pyplot as plt
import numpy as np
import pyfftw as fftw3f
from pyfftw import empty_aligned
from scipy import ndimage
from scipy.fftpack import fftshift, ifftshift, fftn
from skimage import io

NTHREADS = 1

plt.rcParams['figure.figsize'] = [15, 5]
plt.rcParams['image.cmap'] = 'gray'


class PSF:

    def __init__(self, file_path, resolution):
        self.file_path = file_path
        self.resolution = resolution
        self.psf = io.imread(file_path)

    def adjust_resolution(self, img_resolution):
        (pz, px, py) = self.resolution
        (dz, dx, dy) = img_resolution

        self.psf = ndimage.zoom(self.psf, [pz / dz, px / dx, py / dy])

    def resize(self, chunk_size, overlap):
        overlap = tuple(map((2).__mul__, overlap))
        data_size = tuple(map(sum, zip(chunk_size, overlap)))

        if not self.psf.shape == data_size:
            g_ = empty_aligned(shape=data_size, dtype='complex64')
            H_ = empty_aligned(shape=data_size, dtype='complex64')

            sx, sy, sz = np.array(data_size).astype('float') / self.psf.shape
            OT = fftshift(fftn(fftshift(self.psf)))

            if data_size[2] > 1:
                pr = ndimage.zoom(OT.real, [sx, sy, sz], order=1)
                pi = ndimage.zoom(OT.imag, [sx, sy, sz], order=1)
            else:
                pr = ndimage.zoom(OT.real.squeeze(), [sx, sy], order=1).reshape(data_size)
                pi = ndimage.zoom(OT.imag.squeeze(), [sx, sy], order=1).reshape(data_size)

            H_[:] = ifftshift(pr + 1j * pi)
            func = fftw3f.FFTW(H_, g_, direction='FFTW_BACKWARD', axes=range(H_.ndim), threads=NTHREADS,
                               flags=('FFTW_MEASURE',))
            func.execute()
            self.psf = ifftshift(g_.real).clip(min=0)
            self.psf = self.psf / self.psf.sum()
        else:
            self.psf = self.psf / self.psf.sum()

    def calculate_otf(self):
        FTshape = [int(self.psf.shape[0]), int(self.psf.shape[1]), int(self.psf.shape[2] / 2 + 1)]

        self.psf = self.psf.astype('f4')
        psf2 = 1.0 * self.psf[::-1, ::-1, ::-1]

        # allocate memory
        H = empty_aligned(shape=FTshape, dtype='complex64')
        Ht = empty_aligned(shape=FTshape, dtype='complex64')

        # create plans & calculate OTF and conjugate transformed OTF
        p1 = fftw3f.FFTW(self.psf, H, direction='FFTW_FORWARD', axes=range(self.psf.ndim), flags=('FFTW_MEASURE',),
                         threads=NTHREADS)
        p1.execute()
        p2 = fftw3f.FFTW(psf2, Ht, direction='FFTW_FORWARD', axes=range(psf2.ndim), flags=('FFTW_MEASURE',),
                         threads=NTHREADS)
        p2.execute()
        Ht /= self.psf.size
        H /= self.psf.size

        return H, Ht

    def flip(self):
        self.psf = np.flip(self.psf)

    def visualize(self, title="PSF"):

        max_val = np.max(self.psf)
        max_idx = np.unravel_index(np.argmax(self.psf), self.psf.shape)

        fig = plt.figure()
        a = fig.add_subplot(1, 3, 1)
        plt.imshow(self.psf[:, :, max_idx[2]], clim=(0, max_val))
        a.set_title('Y =' + str(max_idx[2]))

        a = fig.add_subplot(1, 3, 2)
        plt.imshow(self.psf[:, max_idx[1], :], clim=(0, max_val))
        a.set_title('X =' + str(max_idx[1]))

        a = fig.add_subplot(1, 3, 3)
        plt.imshow(self.psf[max_idx[0], :, :], clim=(0, max_val))
        a.set_title('Z =' + str(max_idx[0]))
        fig.suptitle(title)
        fig.show()

        self.info()

    def info(self):
        print("Shape: ", self.psf.shape)
        print("Max: ", np.max(self.psf))
        print("Min: ", np.min(self.psf))
        print("Mean: ", np.mean(self.psf))
