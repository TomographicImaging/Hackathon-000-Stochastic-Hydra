import sirf.STIR as pet
import ImageQualityCallback
from ImageQualityCallback import MSE, MAE, PSNR, ImageQualityCallback
import datatime


class QualityMetricsFactory(object):
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

