class PreconditionerFactory(object):
    def __init__(self,cfg):
        if cfg.name == 'MLEM':
            self.preconditioner = MLEMPreconditioner
        pass

                                    
            
class MLEMPreconditioner:
    def __init__(self,cfg_preconditioning):
        self.delta = cfg_preconditioning.delta
        self.freeze_epoch = cfg_preconditioning.freeze_epoch


    def compute_total_sensitivities(self, dataset, operators):
        ones = dataset.acquisition_data.get_uniform_copy(1)
        self.delta = dataset.reference_image.get_uniform_copy(self.delta)
        acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
        acq_model.set_num_tangential_LORs(10)
        #acq_model.set_acquisition_sensitivity(dataset.multiplicative_factors)
        acq_model.set_matrix(acq_model.get_matrix().set_restrict_to_cylindrical_FOV(False))
        acq_model.set_up(dataset.acquisition_data,self.delta)
        self.total_sensitivities = acq_model.adjoint(ones)
        """ import matplotlib.pyplot as plt
        plt.imshow(self.total_sensitivities.as_array()[0])
        plt.savefig('ZELJKO') """
        """ for i in range(len(operators)):
            if i == 0:
                self.total_sensitivities = operators[i].adjoint(ones)
            else:
                self.total_sensitivities += operators[i].adjoint(ones) """

    def __call__(self):
        return lambda i, x: (x+self.delta).divide(self.total_sensitivities)