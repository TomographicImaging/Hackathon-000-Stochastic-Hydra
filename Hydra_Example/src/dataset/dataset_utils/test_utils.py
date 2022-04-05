

""" def PET_subsetoperator(num_subsets, lor, sinogram, image,)
    # Loop over physical subsets
    for k in range(num_subsets):
        # Set up
        acq_models[k].set_num_tangential_LORs(lor)
        acq_models[k].set_acquisition_sensitivity(asm)
        self.acq_models[k].set_up(sinogram, image_template)    
        self.acq_models[k].num_subsets = num_subsets
        self.acq_models[k].subset_num = k 

        # compute masks 
        mask = acq_models[k].direct(im_one)
        masks.append(mask)

    data_fits = [KullbackLeibler(b=sinogram, eta=addfact, mask=mask.as_array(), use_numba=True) for mask in masks] """