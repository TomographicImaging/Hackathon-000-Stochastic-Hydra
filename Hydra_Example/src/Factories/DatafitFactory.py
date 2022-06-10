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
from cil.optimisation.functions import KullbackLeibler, OperatorCompositionFunction, BlockFunction
from cil.optimisation.operators import \
    CompositionOperator, BlockOperator, LinearOperator, GradientOperator, ScaledOperator
from cil.plugins.ccpi_regularisation.functions import FGP_TV


cil_path = '/home/jovyan/Hackathon-000-Stochastic-Algorithms/cil/'
sys.path.append(cil_path)

class DatafitFactory(object):
    def __init__(self,cfg):
        self.cfg=cfg
                
    def __call__(self,dataset,acquisition_model,masks):
                
        if self.cfg.modality.functionals.datafit.KL:
            
            # SPDHG
            if self.cfg.modality.algorithm.name == "SPDHG":
                datafits = [KullbackLeibler(b=dataset.acquisition_data, 
                                             eta=dataset.additive_factors, 
                                             mask=mask.as_array(), 
                                             use_numba=True) for mask in masks]
                
            # gradient-based algorithms
            else:
                kls = [KullbackLeibler(b=dataset.acquisition_data, 
                                     eta=dataset.additive_factors, 
                                     mask=mask.as_array(), 
                                     use_numba=True) for mask in masks]
                datafits = [OperatorCompositionFunction(f,am) for f,am in zip(kls,acquisition_model)]
                
            # block components together
            datafit = BlockFunction(*datafits)
            return datafit
            
            

        
                