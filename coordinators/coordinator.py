import omegaconf
from omegaconf import DictConfig, OmegaConf
import hydra
import sirf.STIR as pet
msg = pet.MessageRedirector(info=None, warn=None, errr=None)
pet.set_verbosity(0)
pet.AcquisitionData.set_storage_scheme("memory")

import numpy as np

from hydrasrc.Classes.Dataset import Dataset
from hydrasrc.Factories.AcquisitionModelFactory import AcquisitionModelFactory
from hydrasrc.Factories.PriorFactory import PriorFactory
from hydrasrc.Factories.DatafitFactory import DatafitFactory
from hydrasrc.Factories.AlgorithmFactory import AlgorithmFactory
from hydrasrc.Factories.QualityMetricsFactory import QualityMetricsFactory

@hydra.main(config_path="../cfgs", config_name="defaults")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # l object = r class(init: cfg)
    dataset = Dataset(cfg)
    # dataset must contain: acq data, mutl and add, warm start, ref image, roi mask dictionary

    # l object (pointing to classes) = r factory for classes
    acquisition_model_class = AcquisitionModelFactory(cfg)
    prior_class = PriorFactory(cfg)
    datafit_class = DatafitFactory(cfg)
    algorithm_class = AlgorithmFactory(cfg)
    quality_metrics_class = QualityMetricsFactory(cfg)

    # Want to get to the point of no MASKS
    acquisition_model, masks = acquisition_model_class(dataset)
    prior = prior_class()
    datafit = datafit_class(dataset,acquisition_model,masks)
    quality_metrics = quality_metrics_class(dataset)
    algorithm, num_iterations = algorithm_class(dataset, datafit, prior, acquisition_model)
    
    algorithm.run(num_iterations, callback=quality_metrics.eval)


if __name__ == "__main__":
    main()

