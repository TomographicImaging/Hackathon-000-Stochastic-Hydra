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

class PriorFactory(object):
    def __init__(self,cfg):
        if cfg.algorithm.prior.name == "no_prior":
            self.prior = self.no_prior()
        if cfg.algorithm.prior.name == "non_negativity":
            self.prior = self.non_negativity()

    def __call__(self):
        return self.prior

    def no_prior(self):
        return 

    def non_negativity(self):
        return IndicatorBox(lower=0)
