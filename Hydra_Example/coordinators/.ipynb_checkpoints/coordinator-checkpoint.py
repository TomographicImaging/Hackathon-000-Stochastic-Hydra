import omegaconf
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

source_path = '/home/jovyan/Hackathon-000-Stochastic-Hydra/Hydra_Example/src'
sys.path.append(source_path)


from Classes.Dataset import Dataset
from Classes.AcquisitionModel import AcquisitionModel
from Classes.Prior import Prior
from Factories.DataFitFactory import DataFitFactory
from Factories.AlgorithmFactory import AlgorithmFactory

# utils_path = '/home/jovyan/Hackathon/Hackathon-000-Stochastic-Hydra/Hydra_Example/src/utils.py'
# runpy.run_path(utils_path)
# import utils
# from utils import set_up_acquisition_model_with_data
def set_up_acquisition_model_with_data(acquisition_model, dataset):
    # acquisition_model.set_acquisition_sensitivity(dataset.multiplicative_factor)
    acquisition_model.set_up(dataset.acq_data, dataset.image_template)  

import subprocess

try:
    subprocess.check_output('nvidia-smi')
    print('Nvidia GPU detected!')
    device = 'gpu'
except Exception: # this command not being found can raise quite a few different errors depending on the configuration
    device = 'cpu'
    print('No Nvidia GPU in system!')
    




@hydra.main(config_path="../cfgs", config_name="defaults")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


    # l object = r class(init: cfg)
    dataset = Dataset(cfg)
    acquisition_model = AcquisitionModel(cfg).acquisition_model
    # warm_start_image = WarmStartImage(cfg)
    # ground_truth = GroundTruth(cfg)
    prior = Prior(cfg)
    
    # l object (pointing to classes) = r factory for classes
    datafitfactory = DataFitFactory(cfg)
    algorithmfactory = AlgorithmFactory(cfg)
    # qualitymetricsfactory = QualityMetricsFactory(cfg)
    
    
    set_up_acquisition_model_with_data(acquisition_model, dataset)
    datafitfactory.set_up_data_fit(dataset,acquisition_model)
    # quality_metrics = qualitymetricsfactory(groundtruth)
    # algorithm = algorithmfactory(dataset, datafit, prior, acquisition_model, quality_metrics, warm_start_image)
    algorithmfactory.set_up_algorithm(dataset, datafitfactory, prior, acquisition_model, quality_metrics=None, warm_start_image=None)
    
    algorithmfactory.algorithm.run(10)
    where_to_save = '/home/jovyan/Hackathon/Hackathon-000-Stochastic-Hydra/Hydra_Example/results'
    algorithmfactory.save_solution(where_to_save)


if __name__ == "__main__":
    main()