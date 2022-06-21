import sirf.STIR as pet
from ImageQualityCallback import ImageQualityCallback
from datetime import datetime
import tensorboardX
import numpy as np


class QualityMetricsFactory(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.metrics_dict = {'MSE':self.MSE, 'MAE':self.MAE, 'PSNR':self.PSNR}
        self.statistics_dict = {'MEAN': (lambda x: x.mean()),
                                    'STDDEV': (lambda x: x.std())}
        dt_string = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_summary_writer = tensorboardX.SummaryWriter(f'exp-{dt_string}')

    def __call__(self, dataset):

        return ImageQualityCallback(
            dataset.reference_image, 
            self.tb_summary_writer, 
            roi_mask_dict = dataset.roi_mask_dict, 
            metrics_dict = self.metrics_dict, 
            statistics_dict = self.statistics_dict)

    def MSE(self, x,y):
        """ mean squared error between two numpy arrays
        """
        return ((x-y)**2).mean()

    def MAE(self, x,y):
        """ mean absolute error between two numpy arrays
        """
        return np.abs(x-y).mean()

    def PSNR(self, x, y, scale = None):
        """ peak signal to noise ratio between two numpy arrays x and y
            y is considered to be the reference array and the default scale
            needed for the PSNR is assumed to be the max of this array
        """
    
        mse = ((x-y)**2).mean()
    
        if scale == None:
            scale = y.max()
    
        return 10*np.log10((scale**2) / mse)

