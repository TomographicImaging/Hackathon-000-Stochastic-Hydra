import sys
import numpy as np
import cil
import tensorboardX
from scipy.ndimage import binary_dilation
from datetime import datetime

quality_metrics_path = '/home/jovyan/Hackathon-000-Stochastic-QualityMetrics/'
sys.path.append(quality_metrics_path)
from ImageQualityCallback import MSE, MAE, PSNR, ImageQualityCallback


class QualityMetrics(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.metrics_dict = {'MSE':MSE, 'MAE':MAE, 'PSNR':PSNR}
        self.statistics_dict = {'MEAN': (lambda x: x.mean()),
                                    'STDDEV': (lambda x: x.std())}
        dt_string = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_summary_writer = tensorboardX.SummaryWriter(f'runs/exp-{dt_string}')

    def __call__(self, dataset):

        return ImageQualityCallback(
            dataset.reference_image, 
            self.tb_summary_writer, 
            roi_mask_dict = dataset.roi_mask_dict, 
            metrics_dict = self.metrics_dict, 
            statistics_dict = self.statistics_dict)

