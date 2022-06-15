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


cil_path = '/home/jovyan/Hackathon-000-Stochastic-Algorithms/cil/'
sys.path.append(cil_path)

from sirf.Utilities import examples_data_path


class Dataset(object):
    def __init__(self,cfg):
        self.cfg=cfg

        if cfg.modality.dataset.name == 'folder':
            folder = cfg.modality.dataset.path
            self.reference_image = pet.ImageData('{}/reference_image.hv'.format(folder))
            self.acquisition_data = pet.AcquisitionData('{}/acquisition_data.hs'.format(folder))
            mf = pet.AcquisitionData('{}/multiplicative_factors.hs'.format(folder))
            self.multiplicative_factors = pet.AcquisitionSensitivityModel(mf)
            self.additive_factors = pet.AcquisitionData('{}/additive_factors'.format(folder))


        
        if cfg.modality.dataset.name == 'test':
            gt = pet.ImageData('{}/{}/{}'.format(
                examples_data_path("PET"),
                'thorax_single_slice',
                'emission.hv'))
            attn_img = pet.ImageData('{}/{}/{}'.format(
                examples_data_path("PET"),
                'thorax_single_slice',
                'attenuation.hv'))
            acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
            data_template = pet.AcquisitionData('{}/{}/{}'.format(
                examples_data_path("PET"),
                'thorax_single_slice',
                'template_sinogram.hs'))            
            acq_model.set_up(data_template, gt)
            self.acquisition_data = acq_model.forward(gt)
            multiplicative_factors = self.acquisition_data.clone().fill(1.0)
            self.multiplicative_factors = pet.AcquisitionSensitivityModel(multiplicative_factors)
            self.additive_factors = self.acquisition_data.clone()
            self.additive_factors.fill(0.01)
            self.reference_image = gt                                                       
            self.image_template = gt.clone().fill(1.0)
            self.warm_start_image = None

            # Create ROIs
            # threshold the numpy array behind the image
            image_array = self.reference_image.as_array()

            # warning: ROI images have dtype float32, but should better be uint8
            # lesion ROI
            roi1_image = self.reference_image.copy()
            roi1_image.fill(image_array > (0.4*image_array.max()))

            # dilated lesion ROI
            roi2_image = self.reference_image.copy()
            roi2_image.fill(binary_dilation(roi1_image.as_array()))

            # "everthing but the background" ROI
            roi3_image = self.reference_image.copy()
            roi3_image.fill(image_array > (0.05*image_array.max()))
            
            self.roi_mask_dict = {'roi1':roi1_image,'roi2':roi2_image,'roi3':roi3_image}

                                    
            
