import sirf.STIR as pet
from cil.framework import BlockDataContainer, ImageGeometry, BlockGeometry
from cil.optimisation.algorithms import PDHG, SPDHG
from cil.optimisation.functions import KullbackLeibler

from algorithmsrc.NewFISTA import ISTA, FISTA
from algorithmsrc.NewSubsetSumFunction import SAGAFunction

from ..utils import get_tau, get_sigmas

class AlgorithmFactory(object):
    def __init__(self,cfg):
        if cfg.name == 'SPDHG':
            algorithm = SPDHGAlgorithm(cfg)
        elif cfg.name == 'ISTA':
            algorithm = ISTAAlgorithm(cfg)
        elif cfg.name == 'asd':
            raise NotImplementedError
        elif cfg.name == 'as':
            raise NotImplementedError
        elif cfg.name == 'das':
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        self.algorithm = algorithm

    def __call__(self, dataset, datafit, prior, acquisition_model):
        return self.algorithm(dataset, datafit, prior, acquisition_model)





class SPDHGAlgorithm(object):
    def __init__(self,cfg):
        self.primal_dual_balance = cfg.parameters.primal_dual_balance
        self.num_subsets = cfg.parameters.num_subsets
        self.preconditioning = False
        
        if cfg.subset_sampling.name == "uniform":
            self.subset_probability = [1/self.num_subsets] * self.num_subsets
            self.num_iterations = cfg.parameters.num_epochs * self.num_subsets
        elif cfg.name == 'a':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def __call__(self, dataset, datafit, prior, acquisition_model):
        initial = dataset.reference_image.get_uniform_copy(0)
        gamma = self.primal_dual_balance
        if self.preconditioning:
            # compute preconditioned step-sizes
            tau = 1/gamma * get_tau(acquisition_model, prob)
            sigma = gamma * get_sigmas(acquisition_model)
            
            return SPDHG(            
                    f=datafit, 
                    g=prior, 
                    operator=acquisition_model,
                    tau=tau,
                    sigma=sigma,
                    prob=self.subset_probability,
                    initial=initial,
                    max_iteration=1e10,         
                    update_objective_interval=1,
                    use_axpby=False
                    ), self.num_iterations
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
                    prob=self.subset_probability,
                    initial=initial,
                    norms=None,
                    max_iteration=1e10,         
                    update_objective_interval=1,
                    use_axpby=False
                    ), self.num_iterations



class ISTAAlgorithm(object):
    def __init__(self,cfg):
        self.num_subsets = cfg.parameters.num_subsets

    def __call__(self, dataset, datafit, prior, acquisition_model,
                                preconditioner,
                                step_size,
                                subset_gradient,
                                warm_start):
        if self.preconditioner_name == 'MLEM':
            self.preconditioner.compute_total_sensitivities(dataset, acquisition_model)
        
        return ISTA(
                initial = dataset.reference_image.get_uniform_copy(0),  
                f=subset_gradient(datafit,precond=self.preconditioner()),
                g=prior, 
                step_size=step_size,
                update_objective_interval=self.num_subsets,
                max_iteration=1e10, 
                check_convergence_criterion=False
                ), self.num_iterations
