import sirf.STIR as pet

from calendar import day_abbr
import os
import runpy
import time
import logging
from datetime import datetime
import argparse
import numpy as np
import scipy.io
from functools import partial
from numbers import Number
import warnings
import numpy
from numpy.linalg import norm
import os
import sys
import shutil
#import scipy
#from scipy import optimize
import sirf.STIR as pet
from sirf.Utilities import examples_data_path


import sirf.STIR as pet
pet.set_verbosity(1)

class dataset_test(object):

    def __init__(self, name):

        data_path = os.path.join(examples_data_path('PET'), 'thorax_single_slice')

        image = pet.ImageData(os.path.join(data_path, 'emission.hv'))
        attn_image = pet.ImageData(os.path.join(data_path, 'attenuation.hv'))


        acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
        template = pet.AcquisitionData(os.path.join(data_path, 'template_sinogram.hs'))
        acq_model.set_up(template, image)
        acquired_data = acq_model.forward(image)
        background_term = acquired_data.get_uniform_copy(acquired_data.max()/10)
        acq_model.set_background_term(background_term)
        self.acquired_data = acq_model.forward(image)
        self.acq_model = acq_model
        self.initial = image.get_uniform_copy(0)

    def get_data(self):
        return self.acquired_data
    def get_initial(self):
        return self.initial
    def get_acq_model(self):
        return self.acq_model
    def get_callbacks(self):
        return None


