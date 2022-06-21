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

from scipy.ndimage import binary_dilation

from sirf.Utilities import examples_data_path


class Dataset(object):
    def __init__(self,cfg):

        if cfg.dataset.modality == 'PET':
            self.reference_image = pet.ImageData(cfg.dataset.data.reference)
            self.acquisition_data = pet.AcquisitionData(cfg.dataset.data.prompts)
            mf = pet.AcquisitionData(cfg.dataset.data.multiplicative)
            self.multiplicative_factors = pet.AcquisitionSensitivityModel(mf)
            self.additive_factors = pet.AcquisitionData(cfg.dataset.data.additive)
            self.roi_mask_dict = {}
            for i in range(10):
                if 'ROI{}'.format(i) in cfg.dataset.keys():
                    self.roi_mask_dict[cfg.dataset['ROI{}'.format(i)].name]  = pet.ImageData(cfg.dataset['ROI{}'.format(i)].path)
        else:
            raise NotImplementedError

                                    
            
