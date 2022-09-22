from hydrasrc.Factories.SubsetSamplingFactory import SubsetSamplingFactory

class SubsetGradientFactory(object):
    def __init__(self,cfg):
        self.subset_sampling = SubsetSamplingFactory(cfg.subset_sampling) 
        pass

                                    
            
