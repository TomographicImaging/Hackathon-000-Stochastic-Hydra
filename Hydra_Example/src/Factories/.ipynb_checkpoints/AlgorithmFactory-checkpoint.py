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
from NewFISTA import ISTA

class AlgorithmFactory(object):
    def __init__(self,cfg):
        self.cfg=cfg        
                
    def set_up_algorithm(self, dataset, datafit, prior, acquisition_model, quality_metrics, warm_start_image):
        if self.cfg.modality.acq_model.num_subsets == 1 and self.cfg.modality.algorithm.name == "GD":
            initial = dataset.image_template.get_uniform_copy(0)
            step_size = self.cfg.modality.algorithm.stepsize
            self.algorithm = ISTA(initial=initial, f=datafit.datafit, g=prior.prior,
                step_size=step_size, max_iteration=1e10,check_convergence_criterion=False)
            
    def save_solution(self,where_to_save):
        self.algorithm.solution.write('{}/{}'.format(where_to_save,'solution'))
        np.save('{}/{}'.format(where_to_save,'objective'),self.algorithm.objective)
        
        