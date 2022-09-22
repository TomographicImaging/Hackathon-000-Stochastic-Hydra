import sirf.STIR as pet
from cil.framework import BlockDataContainer, ImageGeometry, BlockGeometry
from cil.optimisation.algorithms import PDHG, SPDHG
from cil.optimisation.functions import KullbackLeibler

from algorithmsrc.NewFISTA import ISTA, FISTA
from algorithmsrc.NewSubsetSumFunction import SAGAFunction


from hydrasrc.Factories.AlgorithmFactory import AlgorithmFactory
from hydrasrc.Factories.PreconditionerFactory import PreconditionerFactory
from hydrasrc.Factories.PriorFactory import PriorFactory
from hydrasrc.Factories.StepSizeFactory import StepSizeFactory
from hydrasrc.Factories.SubsetGradientFactory import SubsetGradientFactory
from hydrasrc.Factories.WarmStartFactory import WarmStartFactory

from ..utils import get_tau, get_sigmas

class AlgorithmConfigurationFactory(object):
    def __init__(self,cfg):

        self.preconditioner = PreconditionerFactory(cfg.preconditioner)
        self.step_size = StepSizeFactory(cfg.step_size)
    
        self.prior = PriorFactory(cfg.prior)
        self.subset_gradient = SubsetGradientFactory(cfg.subset_gradient) 
        self.warm_start = WarmStartFactory(cfg.warm_start) 
             
        self.algorithm = AlgorithmFactory(cfg.algorithm)

    def __call__(self, dataset, datafit, prior, acquisition_model):
        self.preconditioner(dataset, datafit, prior, acquisition_model)
        self.step_size(dataset, datafit, prior, acquisition_model)
        self.subset_gradient(dataset, datafit, prior, acquisition_model, self.subset_sampling)
        self.warm_start(dataset, datafit, prior, acquisition_model)
        return self.algorithm(dataset, datafit, prior, acquisition_model,
                                self.preconditioner,
                                self.step_size,
                                self.subset_gradient,
                                self.warm_start)






