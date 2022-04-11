

def set_up_acquisition_model_with_data(acquisition_model, dataset):
    if dataset.multiplicative_factors is not None:
        acquisition_model.set_acquisition_sensitivity(dataset.multiplicative_factors)
    acquisition_model.set_up(dataset.acquisition_data, dataset.image_template)  
