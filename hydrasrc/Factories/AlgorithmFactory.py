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


from algorithmsrc.NewFISTA import ISTA, FISTA

from ..utils import get_tau, get_sigmas

class AlgorithmFactory(object):
    def __init__(self,cfg):
        self.cfg=cfg        
                
    def __call__(self, dataset, datafit, prior, acquisition_model):
        
        initial = dataset.image_template.get_uniform_copy(0)
        
        if self.cfg.modality.algorithm.name == "GD":
            
            step_size = self.cfg.modality.algorithm.stepsize
            num_iterations = self.cfg.modality.algorithm.num_epochs
            
            return ISTA(
                    initial=initial, 
                    f=datafit, 
                    g=prior,
                    step_size=step_size, 
                    max_iteration=1e10,
                    check_convergence_criterion=False), num_iterations
        
        elif self.cfg.modality.algorithm.name == "SPDHG":
            
            # primal-dual balance
            gamma = self.cfg.modality.algorithm.hyperparameters.primal_dual_balance
            
            # number of subsets
            num_subsets = self.cfg.modality.acq_model.num_subsets
            
            # probabilities and number of iterations
            if self.cfg.modality.algorithm.sampling == "uniform":
                prob = [1/num_subsets] * num_subsets
                num_iterations = self.cfg.modality.algorithm.num_epochs * num_subsets
            
            if self.cfg.modality.algorithm.preconditioning:
                
                # compute preconditioned step-sizes
                tau = 1/gamma * get_tau(acquisition_model, prob)
                sigma = gamma * get_sigmas(acquisition_model)
                
                return SPDHG(            
                        f=datafit, 
                        g=prior, 
                        operator=acquisition_model,
                        tau=tau,
                        sigma=sigma,
                        prob=prob,
                        initial=initial,
                        max_iteration=1e10,         
                        update_objective_interval=1,
                        use_axpby=False
                        ), num_iterations
            else:
                
                # should automatically set-up scalar tau and sigmas
                # by computing the norms of the partial operators
                
                    return SPDHG(            
                        f=datafit, 
                        g=prior, 
                        operator=acquisition_model,
                        tau=None,
                        sigma=None,
                        gamma=gamma,
                        prob=prob,
                        initial=initial,
                        norms=None,
                        max_iteration=1e10,         
                        update_objective_interval=1,
                        use_axpby=False
                        ), num_iterations
            

        
        