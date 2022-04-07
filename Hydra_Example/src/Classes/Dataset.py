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


cil_path = '/home/jovyan/Hackathon-000-Stochastic-Algorithms/cil/'
sys.path.append(cil_path)

from sirf.Utilities import examples_data_path

class Dataset(object):
    def __init__(self,cfg):
        self.cfg=cfg
        
        if cfg.modality.dataset.name == 'test':
            gt = pet.ImageData('{}/{}/{}'.format(
                examples_data_path("PET"),
                'thorax_single_slice',
                'emission.hv'))
            attn_img = pet.ImageData('{}/{}/{}'.format(
                examples_data_path("PET"),
                'thorax_single_slice',
                'attenuation.hv'))
            data_path = os.path.join(examples_data_path('PET'), 'thorax_single_slice')
            acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
            template = pet.AcquisitionData(os.path.join(data_path, 'template_sinogram.hs'))
            acq_model.set_up(template, gt)
            self.acq_data = acq_model.forward(gt)
            self.multiplicative_factor = self.acq_data.clone().fill(1.0)
            self.additive_factor = self.acq_data.clone()
            self.additive_factor.fill(0.01)
                                                       
        self.image_template = self.acq_data.create_uniform_image(0.0)
                                    
            
