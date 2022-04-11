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

    def __call__(self):
        
        if self.cfg.modality.acq_model.num_subsets == 1:
            acquisition_model = pet.AcquisitionModelUsingRayTracingMatrix()
            acquisition_model.set_num_tangential_LORs(self.cfg.modality.acq_model.LOR)

        return acquisition_model
