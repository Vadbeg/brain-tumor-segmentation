"""Training module"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from monai.losses import DiceLoss
from monai.metrics.meandice import compute_meandice
from torch.utils.data import DataLoader

from brain_tumor_segmentation.modules.data.datasets.train_dataset import BrainSegDataset
from brain_tumor_segmentation.modules.data.utils import (
    create_data_loader,
    get_load_transforms,
    get_train_transforms_3d,
    get_train_val_paths,
    get_val_transforms_3d,
)
from brain_tumor_segmentation.modules.model.unet3d import get_unet3d_model


class BrainSegmentation3DModel(pl.LightningModule):
    def __init__(
        self,
        dataset_folder: Union[str, Path],
        train_split_percent: float = 0.7,
        dataset_item_limit: Optional[int] = 1,
        shuffle_dataset: bool = True,
        image_pattern: str = '**/*_flair.nii.gz',
        mask_pattern: str = '**/*_seg.nii.gz',
        spatial_size: Tuple[int, int, int] = (196, 196, 128),
        batch_size: int = 2,
        num_processes: int = 1,
        learning_rate: float = 0.001,
        in_channels: int = 1,
        out_channels: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.img_key = 'image'
        self.lbl_key = 'mask'

        self.train_paths, self.val_paths = get_train_val_paths(
            dataset_folder=dataset_folder,
            img_key=self.img_key,
            lbl_key=self.lbl_key,
            train_percent=train_split_percent,
            item_limit=dataset_item_limit,
            shuffle=shuffle_dataset,
            image_pattern=image_pattern,
            mask_pattern=mask_pattern,
        )
        self.load_transform = get_load_transforms(
            img_key=self.img_key,
            lbl_key=self.lbl_key,
            orig_labels=[1, 2, 4],
            target_labels=[1, 2, 3],
            spatial_size=spatial_size,
        )

        self.batch_size = batch_size
        self.num_processes = num_processes
        self.learning_rate = learning_rate

        self.loss = DiceLoss()
        self.model = get_unet3d_model(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
        )

    def training_step(
        self, batch: Dict, batch_id: int
    ) -> Dict[str, Any]:  # pylint: disable=W0613
        image = batch[self.img_key]
        label = batch[self.lbl_key]

        result = self.model(image)
        loss = self.loss(result, label)

        self.log(
            name='train_loss',
            value=loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        self._log_metrics(preds=result, target=label, prefix='train')

        return {'loss': loss, 'pred': result, 'label': label}

    def validation_step(
        self, batch: Dict, batch_id: int
    ) -> Dict[str, Any]:  # pylint: disable=W0613
        image = batch[self.img_key]
        label = batch[self.lbl_key]

        result = self.model(image)
        loss = self.loss(result, label)

        self.log(
            name='val_loss',
            value=loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        self._log_metrics(preds=result, target=label, prefix='val')

        return {'loss': loss, 'pred': result, 'label': label}

    def train_dataloader(self) -> DataLoader:
        train_brain_dataset = BrainSegDataset(
            image_mask_paths=self.train_paths, load_transforms=self.load_transform
        )

        train_brain_dataloader = create_data_loader(
            dataset=train_brain_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_processes,
        )

        return train_brain_dataloader

    def val_dataloader(self) -> DataLoader:
        val_brain_dataset = BrainSegDataset(
            image_mask_paths=self.val_paths, load_transforms=self.load_transform
        )

        val_brain_dataloader = create_data_loader(
            dataset=val_brain_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_processes,
        )

        return val_brain_dataloader

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, factor=0.5, patience=3, mode='min'
        )

        configuration = {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'},
        }

        return configuration

    def _log_metrics(
        self, preds: torch.Tensor, target: torch.Tensor, prefix: str
    ) -> None:
        dice_value = self._calculate_dice(preds=preds, target=target)

        self.log(
            name=f'{prefix}_f1',
            value=dice_value,
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )

    @staticmethod
    def _calculate_dice(preds: torch.Tensor, target: torch.Tensor) -> float:
        dice_value = compute_meandice(
            y_pred=preds.cpu(),
            y=target.cpu(),
        )

        dice_value = torch.mean(dice_value).detach().cpu().numpy()
        dice_value = float(dice_value)

        return dice_value
