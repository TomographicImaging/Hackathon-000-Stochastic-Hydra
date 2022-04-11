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
from src.Factories import QualityMetricsFactory
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

source_path = '/home/jovyan/Hackathon-000-Stochastic-Hydra/Hydra_Example/src/'
sys.path.append(source_path)



from Classes.Dataset import Dataset
from Classes.QualityMetrics import QualityMetrics
from Factories.AcquisitionModelFactory import AcquisitionModelFactory
from Factories.PriorFactory import PriorFactory
from Factories.DatafitFactory import DatafitFactory
from Factories.AlgorithmFactory import AlgorithmFactory
from utils import set_up_acquisition_model_with_data

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
    # dataset must contain: acq data, mutl and add, warm start, ref image, roi mask dictionary

    # l object (pointing to classes) = r factory for classes
    acquisition_model_factory = AcquisitionModelFactory(cfg)
    prior_factory = PriorFactory(cfg)
    datafit_factory = DatafitFactory(cfg)
    algorithm_factory = AlgorithmFactory(cfg)
    quality_metrics_factory = QualityMetricsFactory(cfg)

    
    acquisition_model = acquisition_model_factory()
    set_up_acquisition_model_with_data(acquisition_model, dataset)
    prior = prior_factory()
    datafit = datafit_factory(dataset)
    quality_metrics = quality_metrics_factory(dataset)
    algorithm = algorithm_factory(dataset, datafit, prior, acquisition_model)
    
    algorithm.run(10, callback=quality_metrics.eval)

    where_to_save = "../results"
    algorithm.solution.write('{}/{}'.format(where_to_save,'solution'))
    np.save('{}/{}'.format(where_to_save,'objective'),algorithm.objective)


if __name__ == "__main__":
    main()

