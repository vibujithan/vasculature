#!/usr/bin/python

###############
# fftwWisdom.py
#
# Copyright David Baddeley, 2012
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
################
import os
import pickle

import pyfftw

WISDOMFILE = os.path.join(os.path.split(__file__)[0], 'fftw_wisdom.pkl')


def load_wisdom():
    if os.path.exists(WISDOMFILE):
        with open(WISDOMFILE, 'rb') as f:
            pyfftw.import_wisdom(pickle.load(f))


def save_wisdom():
    with open(WISDOMFILE, 'wb') as f:
        pickle.dump(pyfftw.export_wisdom(), f)
