import sirf.STIR as pet
from cil.optimisation.algorithms import PDHG, SPDHG
from cil.optimisation.functions import KullbackLeibler, OperatorCompositionFunction, BlockFunction

class DatafitFactory(object):
    def __init__(self,cfg):
        if cfg.dataset.modality == 'PET':
            datafit = PETDatafit(cfg)
        if cfg.dataset.modality == 'CT':
            datafit = PETDatafit(cfg)
        self.datafit = datafit

    def __call__(self,dataset,acquisition_model,masks):
        return self.datafit(dataset,acquisition_model,masks)



class PETDatafit(object):
    def __init__(self,cfg):
        self.algorithm = cfg.algorithm.name
        self.num_subsets = cfg.algorithm.parameters.num_subsets

    def __call__(self,dataset,acquisition_model,masks):
                
            
        # SPDHG
        if self.algorithm == 'SPDHG':
            datafits = [KullbackLeibler(b=dataset.acquisition_data, 
                                            eta=dataset.additive_factors, 
                                            mask=mask.as_array(), 
                                            use_numba=True) for mask in masks]
            
        # gradient-based algorithms
        else:
            kls = [KullbackLeibler(b=dataset.acquisition_data, 
                                    eta=dataset.additive_factors, 
                                    mask=mask.as_array(), 
                                    use_numba=True) for mask in masks]
            datafits = [OperatorCompositionFunction(f,am) for f,am in zip(kls,acquisition_model)]
            
        # block components together
        datafit = BlockFunction(*datafits)
        return datafit
        
            

        
                