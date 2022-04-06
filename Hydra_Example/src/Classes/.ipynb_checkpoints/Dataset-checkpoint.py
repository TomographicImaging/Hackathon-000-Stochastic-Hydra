from omegaconf import DictConfig, OmegaConf
import hydra

import os
import sys
import runpy
import numpy as np
from functools import partial
from numbers import Number
import warnings

import sirf.STIR as pet
pet.set_verbosity(0)
import sirf.Reg as reg
from cil.framework import BlockDataContainer, ImageGeometry, BlockGeometry
from cil.optimisation.algorithms import PDHG, SPDHG
from cil.optimisation.functions import \
    KullbackLeibler, BlockFunction, IndicatorBox, MixedL21Norm, ScaledFunction, TotalVariation
from cil.optimisation.operators import \
    CompositionOperator, BlockOperator, LinearOperator, GradientOperator, ScaledOperator
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from ccpi.filters import 

cil_path = '/home/jovyan/Hackathon-000-Stochastic-Algorithms/cil/'
sys.path.append(cil_path)

from sirf.Utilities import examples_data_path

class Dataset(object):
    def __init__(self,cfg):
        self.cfg=cfg
        
        if cfg.dataset.name = 'test':
            self.acq_data = pet.ImageData('{}/{}/{}'.format(
                examples_data_path("PET"),
                'thorax_single_slice',
                'emission.hv')
            self.multiplicative_factor = pet.ImageData('{}/{}/{}'.format(
                examples_data_path("PET"),
                'thorax_single_slice',
                'attenuation.hv')
            self.additive_factor = self.acq_data.clone()
            self.additive_factor.fill(0.01*self.acq_data.to_numpy())
                                                       
        self.image_template = self.acq_data.create_uniform_image(0.0)
                                    
            
