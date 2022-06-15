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

class AcquisitionModelFactory(object):
    def __init__(self,cfg):
        self.cfg=cfg


    def __call__(self,dataset):
        
        # number of subsets
        num_subsets = self.cfg.modality.acq_model.num_subsets
        # create acquisition models
        acq_models = [pet.AcquisitionModelUsingRayTracingMatrix() for k in range(num_subsets)]
        # create masks
        im_one = dataset.image_template.clone()
        im_one.fill(1.)
        masks = []

        # Loop over physical subsets
        for k in range(num_subsets):
            # Set up
            acq_models[k].set_num_tangential_LORs(self.cfg.modality.acq_model.LOR)
            acq_models[k].set_acquisition_sensitivity(dataset.multiplicative_factors)
            acq_models[k].set_up(dataset.acquisition_data, dataset.image_template)    
            acq_models[k].num_subsets = num_subsets
            acq_models[k].subset_num = k 

            # compute masks 
            mask = acq_models[k].direct(im_one)
            masks.append(mask)
        
        return BlockOperator(*acq_models), masks
            
# 
#     def __call__(self):
        
#         if self.cfg.modality.acq_model.num_subsets == 1:
#             acquisition_model = pet.AcquisitionModelUsingRayTracingMatrix()
#             acquisition_model.set_num_tangential_LORs(self.cfg.modality.acq_model.LOR)

#         return acquisition_model