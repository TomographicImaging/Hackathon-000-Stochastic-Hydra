cil_path = '/home/jovyan/Hackathon-000-Stochastic-QualityMetrics/'
import sys
sys.path.append(cil_path)
import numpy as np
import cil
import tensorboardX
from ImageQualityCallback import MSE, MAE, PSNR, ImageQualityCallback

class QualityMetrics(object):
    def __init__(self, cfg, reference_image = None, roi_mask = None):
        metrics_dict = {'MSE':MSE, 'MAE':MAE, 'PSNR':PSNR}
        statistics_dict = {'MEAN': (lambda x: x.mean()),
                                    'STDDEV': (lambda x: x.std())}
        from datetime import datetime
        dt_string = datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_summary_writer = tensorboardX.SummaryWriter(f'runs/exp-{dt_string}')
        self.callback = ImageQualityCallback(reference_image, 
            tb_summary_writer, roi_mask_dict = masks, metrics_dict = metrics_dict, statistics_dict = statistics_dict)