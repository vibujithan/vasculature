#!/usr/bin/python
##################
# richardsonLucy.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

from . import fftwWisdom, deconv_mappings as dec

fftwWisdom.load_wisdom()
import numpy as np


class RichardsonLucyDeconvolution(object):
    """Deconvolution class, implementing a variant of the Richardson-Lucy algorithm.

    Derived classed should additionally define the following methods:
    AFunc - the forward mapping (computes Af)
    AHFunc - conjugate transpose of forward mapping (computes \bar{A}^T f)
    LFunc - the likelihood function
    LHFunc - conj. transpose of likelihood function

    see dec_conv for an implementation of conventional image deconvolution with a
    measured, spatially invariant PSF
    """

    def __init__(self) -> object:
        pass

    def startGuess(self, data):
        """starting guess for deconvolution - can be overridden in derived classes
        but the data itself is usually a pretty good guess.
        """
        return 0 * data + data.mean()

    def deconvp(self, args):
        """ convenience function for deconvolving in parallel using processing.Pool.map"""
        return self.deconv(*args)
        # return 0

    def deconv(self, data, lamb, num_iters=10, weights=1, bg=0):
        """This is what you actually call to do the deconvolution.
        parameters are:

        data - the raw data
        lamb - the regularisation parameter (ignored - kept for compatibility with ICTM)
        num_iters - number of iterations (note that the convergence is fast when
                    compared to many algorithms - e.g Richardson-Lucy - and the
                    default of 10 will usually already give a reasonable result)


        """
        # remember what shape we are
        self.dataShape = np.shape(data)
        print(self.dataShape)

        if 'prep' in dir(self) and not '_F' in dir(self):
            self.prep()

        if not np.isscalar(weights):
            self.mask = 1 - weights  # > 0
        else:
            self.mask = 1 - np.isfinite(data.ravel())

        # guess a starting estimate for the object
        self.f = self.startGuess(data).ravel() - bg
        self.fs = np.reshape(self.f, (self.shape))

        # make things 1 dimensional
        # self.f = self.f.ravel()
        data = np.ravel(data)

        self.loopcount = 0

        while self.loopcount < num_iters:
            self.loopcount += 1
            # the residuals
            self.res = weights * (data / (self.Afunc(self.f) + 1e-12 + bg)) + self.mask
            # adjustment
            adjFact = self.Ahfunc(self.res)
            fnew = self.f * adjFact
            # set the current estimate to out new estimate
            self.f[:] = fnew
            # print(100*self.loopcount/num_iters)
        return np.real(self.fs)


class dec_conv(RichardsonLucyDeconvolution, dec.ClassicMappingFFTW):
    def __init__(self, *args, **kwargs):
        RichardsonLucyDeconvolution.__init__(self, *args, **kwargs)
