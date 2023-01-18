import numpy as np
import pyfftw as fftw3f
from pyfftw import empty_aligned
from scipy.fftpack import ifftshift

NTHREADS = 4
print('No of threads:', NTHREADS)
from . import fftwWisdom

fftwWisdom.load_wisdom()


class DeconvMappingBase(object):
    """
    Base class for different types of deconvolution. Ultimate deconvolution classes will inherit from both this and a
    a method class (e.g. ICTM Deconvolution)

    Provides implementations of the following:

    AFunc - the forward mapping (computes Af)
    AHFunc - conjugate transpose of forward mapping (computes \bar{A}^T f)
    LFunc - the likelihood function
    LHFunc - conj. transpose of likelihood function

    Also, defines:

    psf_calc - does any pre-computation to get the PSF into a usable form
    prep - allocate memory / setup FFTW plans etc ...

    """

    def Afunc(self, f):
        """ The forward mapping"""
        raise NotImplementedError('Must be over-ridden in derived class')

    def Ahfunc(self, f):
        """ The conjugate transpose of the forward mapping"""
        raise NotImplementedError('Must be over-ridden in derived class')

    def Lfunc(self, f):
        """ The likelihood function (ICTM deconvolution only)"""
        raise NotImplementedError('Must be over-ridden in derived class')

    def Lhfunc(self, f):
        """ The gonjugate transpose of likelihood function (ICTM deconvolution only)"""
        raise NotImplementedError('Must be over-ridden in derived class')

    def psf_calc(self, H, Ht, data_size):
        """
        do any pre-computation on the PSF. e.g. resizing to match the data shape and/or pre-calculating and
        storing the OTF
        """
        raise NotImplementedError('Must be over-ridden in derived class')

    def prep(self):
        """
        Allocate memory and compute FFTW plans etc (this is separate from psf_calc as an aid to distributing Deconvolution
        objects for parallel processing - psf_calc gets called before Deconv objects are passed, prep gets called when
        the deconvolution is run.

        """


class ClassicMappingFFTW(DeconvMappingBase):
    """Classical deconvolution with a stationary PSF using FFTW for convolutions"""

    def prep(self):
        """Precalculate the OTF etc..."""
        # allocate memory
        self._F = empty_aligned(shape=self.FTshape, dtype='complex64')
        self._r = empty_aligned(shape=self.shape, dtype='f4')

        print('Creating plans for FFTs - this might take a while')
        # calculate plans for other ffts
        self._plan_r_F = fftw3f.FFTW(self._r, self._F, direction='FFTW_FORWARD', axes=range(self._r.ndim),
                                     flags=('FFTW_MEASURE',), threads=NTHREADS)
        self._plan_r_F.execute()
        self._plan_F_r = fftw3f.FFTW(self._F, self._r, direction='FFTW_BACKWARD', axes=range(self._F.ndim),
                                     flags=('FFTW_MEASURE',), threads=NTHREADS)
        self._plan_F_r.execute()
        fftwWisdom.save_wisdom()
        print('Done planning')

    def psf_calc(self, H, Ht, data_size):
        """Precalculate the OTF etc..."""

        self.shape = data_size
        self.FTshape = [int(self.shape[0]), int(self.shape[1]), int(self.shape[2] / 2 + 1)]
        self.Ht = Ht
        self.H = H

    def Lfunc(self, f):
        """convolve with an approximate 2nd derivative likelihood operator in 3D.
        i.e. [[[0,0,0][0,1,0][0,0,0]],[[0,1,0][1,-6,1][0,1,0]],[[0,0,0][0,1,0][0,0,0]]]
        """
        # make our data 3D again
        fs = np.reshape(f, (self.height, self.width, self.depth))
        a = -6 * fs

        a[:, :, 0:-1] += fs[:, :, 1:]
        a[:, :, 1:] += fs[:, :, 0:-1]

        a[:, 0:-1, :] += fs[:, 1:, :]
        a[:, 1:, :] += fs[:, 0:-1, :]

        a[0:-1, :, :] += fs[1:, :, :]
        a[1:, :, :] += fs[0:-1, :, :]

        # flatten data again
        return np.ravel(np.cast['f'](a))

    Lhfunc = Lfunc

    def Afunc(self, f):
        """Forward transform - convolve with the PSF"""
        # fs = reshape(f, (self.height, self.width, self.depth))
        self._r[:] = np.reshape(f, (self._r.shape))
        self._plan_r_F.execute()
        self._F *= self.H
        self._plan_F_r.execute()
        return np.ravel(ifftshift(self._r))

    def Ahfunc(self, f):
        """Conjugate transform - convolve with conj. PSF"""
        self._r[:] = np.reshape(f, (self._r.shape))
        self._plan_r_F.execute()
        self._F *= self.Ht
        self._plan_F_r.execute()
        return np.ravel(ifftshift(self._r))
