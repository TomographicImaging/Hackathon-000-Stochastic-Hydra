cil_path = '/home/jovyan/Hackathon-000-Stochastic-QualityMetrics/'
import sys
sys.path.append(cil_path)
import numpy as np
import cil
import tensorboardX
from ImageQualityCallback import MSE, MAE, PSNR, ImageQualityCallback

class QualityMetrics(object):
    def __init__(self, cfg, reference_image = None, roi_mask = None):
        # threshold the numpy array behind the image

        image = reference_image.get_uniform_copy(0.0)
        image_array = image.as_array()

        # warning: ROI images have dtype float32, but should better be uint8
        # lesion ROI
        roi1_image = image.copy()
        roi1_image.fill(image_array > (0.4*image_array.max()))

        # dilated lesion ROI
        roi2_image = image.copy()
        roi2_image.fill(binary_dilation(roi1_image.as_array()))

        # "everthing but the background" ROI
        roi3_image = image.copy()
        roi3_image.fill(image_array > (0.05*image_array.max()))
        
        roi_mask_dict = {'roi1':roi1_image,'roi2':roi2_image,'roi3':roi3_image}
        
        metrics_dict = {'MSE':MSE, 'MAE':MAE, 'PSNR':PSNR}
        statistics_dict = {'MEAN': (lambda x: x.mean()),
                                    'STDDEV': (lambda x: x.std())}
        from datetime import datetime
        dt_string = datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_summary_writer = tensorboardX.SummaryWriter(f'runs/exp-{dt_string}')
        self.callback = ImageQualityCallback(reference_image, 
            tb_summary_writer, roi_mask_dict = masks, metrics_dict = metrics_dict, statistics_dict = statistics_dict)