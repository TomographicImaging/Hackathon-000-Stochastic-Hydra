import sirf.STIR as pet
from cil.optimisation.operators import BlockOperator

class AcquisitionModelFactory(object):
    def __init__(self,cfg):
        if cfg.dataset.modality == 'PET':
            AcquisitionModel = PETAcquisitionModel(cfg)
        else:
            raise NotImplementedError
        self.get_acquisition_model = AcquisitionModel.get_acquisition_model


    def __call__(self,dataset):
        return self.get_acquisition_model(dataset)


class PETAcquisitionModel(object):
    def __init__(self, cfg):
        self.num_subsets = cfg.algo_config.parameters.num_subsets
        self.LOR = cfg.dataset.modelling.LOR

    def get_acquisition_model(self, dataset):
        if self.num_subsets == 1:
            raise NotImplementedError
        elif self.num_subsets > 1:
            return self.subset_acquisition_model(dataset, self.num_subsets, self.LOR)
        else:
            raise NotImplementedError

    # NEEDS TO BE DONE WITHOUT MASKS
    def subset_acquisition_model(self, dataset, num_subsets, LOR):
        # create acquisition models
        acq_models = [pet.AcquisitionModelUsingRayTracingMatrix() for k in range(num_subsets)]
        # create masks
        im_one = dataset.reference_image.clone()
        im_one.fill(1.)
        masks = []

        # Loop over physical subsets
        for k in range(num_subsets):
            # Set up
            acq_models[k].set_num_tangential_LORs(LOR)
            acq_models[k].set_acquisition_sensitivity(dataset.multiplicative_factors)
            acq_models[k].set_up(dataset.acquisition_data, im_one)    
            acq_models[k].num_subsets = num_subsets
            acq_models[k].subset_num = k 

            # compute masks 
            mask = acq_models[k].direct(im_one)
            masks.append(mask)
        return BlockOperator(*acq_models), masks

class CTAcquisitionModel(object):
    def __init__(self, cfg):
        self.num_subsets = cfg.modality.acq_model.num_subsets
        self.LOR = cfg.modality.acq_model.LOR

    def get_acquisition_model(self,dataset):
        if self.num_subsets == 1:
            raise NotImplementedError
        if self.num_subsets > 1:
            raise NotImplementedError