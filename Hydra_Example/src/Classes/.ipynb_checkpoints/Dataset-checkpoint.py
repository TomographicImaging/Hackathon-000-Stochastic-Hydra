import sirf.STIR as pet
from sirf.Utilities import examples_data_path
class Dataset(object):
    def __init__(self,cfg):
        self.cfg=cfg
        
        image = pet.ImageData(examples_data_path('PET') + '/emission.hv')
        image_array = image.as_array()
        image_array *= cfg.data.gt_scl_fct
        image.fill(image_array)
        attn_image = pet.ImageData(
            os.path.join(cfg.data.path, 'attenuation.hv')
            )
        data_template = pet.AcquisitionData(
        os.path.join(cfg.data.path, 'template_sinogram.hs')
        (examples_data_path('PET') + '/grappa2_1rep.h5')