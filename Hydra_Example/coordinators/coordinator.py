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


import subprocess

try:
    subprocess.check_output('nvidia-smi')
    print('Nvidia GPU detected!')
    device = 'gpu'
except Exception: # this command not being found can raise quite a few different errors depending on the configuration
    device = 'cpu'
    print('No Nvidia GPU in system!')
    
#from utils import set_up_acquisition_model_with_data



@hydra.main(config_path="../cfgs", config_name="defaults")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


    # l object = r class(init: cfg)
    dataset = Dataset(cfg)
    acquisition_model = AcquisitionModel(cfg)
    warm_start_image = WarmStartImage(cfg)
    ground_truth = GroundTruth(cfg)
    prior = Prior(cfg)
    
    # l object (pointing to classes) = r factory for classes
    datafitfactory = DataFitFactory(cfg)
    algorithmfactory = AlgorithmFactory(cfg)
    qualitymetricsfactory = QualityMetricsFactory(cfg)
    
    
    set_up_acquisition_model_with_data(acquisition_model, dataset)
    datafit = datafitfactory(dataset,acquisition_model)
    quality_metrics = qualitymetricsfactory(groundtruth)
    algorithm = algorithmfactory(functionals, acquisition_model, quality_metrics, warm_start_image)
    
    algorithm.run()


if __name__ == "__main__":
    main()