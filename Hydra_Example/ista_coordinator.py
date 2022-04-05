from omegaconf import DictConfig, OmegaConf
import hydra
from cil.optimisation.algorithms.FISTA import ISTA
import numpy as np
import sys

import subprocess

try:
    subprocess.check_output('nvidia-smi')
    print('Nvidia GPU detected!')
    device = 'gpu'
except Exception: # this command not being found can raise quite a few different errors depending on the configuration
    device = 'cpu'
    print('No Nvidia GPU in system!')

@hydra.main(config_path="cfgs", config_name="ista_defaults")
def main(cfg: DictConfig) -> None:
    

    sys.path.append("/home/user/StochasticDev/Hydra_Example/src")
    print(OmegaConf.to_yaml(cfg))

    print(cfg.dataset)
    dataset = hydra.utils.instantiate(cfg.dataset)
    # Get the data-fitting functional
    functional_1 = hydra.utils.instantiate(cfg.functional_1) 
    # Get the other functional
    functional_2 = hydra.utils.instantiate(cfg.functional_2)

    step_size = 0.1#1 / functional_1.get_func(dataset.get_data()).L

    ista = ISTA(initial=dataset.get_initial(), f=functional_1.get_func(dataset.get_data()), g=functional_2.get_func(),
                            step_size=step_size, update_objective_interval=1,
                            max_iteration=1e10, callbacks = dataset.get_callbacks())
    ista.run(epoch =100)
    np.save("test", ista)


if __name__ == "__main__":
    main()