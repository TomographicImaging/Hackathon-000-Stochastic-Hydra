from omegaconf import DictConfig, OmegaConf
import hydra

import sys
cil_path = '/home/jovyan/Hackathon-000-Stochastic-Algorithms/cil/'
sys.path.append(cil_path)
import numpy as np
import os


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
    
    # l object (pointing to classes) = r factory for classes
    qualitymetricsfactory = QualityMetricsFactories(cfg)
    functionalsfactory = FunctionalsFactories(cfg)
    algorithmfactory = AlgorithmFactories(cfg)
    
    
    set_up_acquisition_model_with_data(acquisition_model, dataset)
    functionals = functionalsfactory(acquisition_model, dataset)
    quality_metrics = qualitymetricsfactory(groundtruth)
    algorithm = algorithmfactory(functionals, acquisition_model, quality_metrics, warm_start_image)
    
    algorithm.run()


if __name__ == "__main__":
    main()