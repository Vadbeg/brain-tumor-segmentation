"""Module for starting train script"""


import warnings

from pytorch_lightning.utilities.cli import LightningCLI

from brain_tumor_segmentation.modules.train.training import BrainSegmentation3DModel

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    cli = LightningCLI(model_class=BrainSegmentation3DModel, save_config_callback=None)
