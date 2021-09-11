"""Module for starting train script"""


import warnings

from pytorch_lightning.utilities.cli import LightningCLI

from brain_tumor_classification.modules.train.training import BrainClassification3DModel

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    cli = LightningCLI(
        model_class=BrainClassification3DModel, save_config_callback=None
    )
