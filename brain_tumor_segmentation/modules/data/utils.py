"""Module with data utils"""


import random
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from monai.transforms import (
    AddChanneld,
    CastToTyped,
    Compose,
    LoadImaged,
    MapLabelValued,
    Orientationd,
    RandAdjustContrastd,
    RandFlipd,
    RandRotate90d,
    RandRotated,
    RandScaleIntensityd,
    RandZoomd,
    Resized,
    ScaleIntensityRanged,
    SpatialPadd,
)
from torch.utils.data import DataLoader, Dataset


def get_train_val_paths(
    dataset_folder: Union[str, Path],
    img_key: str = 'image',
    lbl_key: str = 'mask',
    image_pattern: str = '**/*_flair.nii.gz',
    mask_pattern: str = '**/*_seg.nii.gz',
    train_percent: float = 0.7,
    shuffle: bool = True,
    item_limit: int = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    all_image_mask_paths = _get_image_mask_paths(
        dataset_folder=dataset_folder,
        img_key=img_key,
        lbl_key=lbl_key,
        image_pattern=image_pattern,
        mask_pattern=mask_pattern,
    )
    if shuffle:
        random.shuffle(all_image_mask_paths)

    edge_value = int(train_percent * len(all_image_mask_paths))

    train_image_mask_paths = all_image_mask_paths[:edge_value]
    val_image_mask_paths = all_image_mask_paths[edge_value:]

    if item_limit:
        train_image_mask_paths = train_image_mask_paths[:item_limit]
        val_image_mask_paths = val_image_mask_paths[:item_limit]

    return train_image_mask_paths, val_image_mask_paths


def _get_image_mask_paths(
    dataset_folder: Union[str, Path],
    img_key: str = 'image',
    lbl_key: str = 'mask',
    image_pattern: str = '**/*_flair.nii.gz',
    mask_pattern: str = '**/*_seg.nii.gz',
) -> List[Dict[str, str]]:
    dataset_folder = Path(dataset_folder)

    image_paths = _get_paths(folder=dataset_folder, pattern=image_pattern)
    mask_paths = _get_paths(folder=dataset_folder, pattern=mask_pattern)

    image_paths.sort()
    mask_paths.sort()

    image_mask_paths = [
        {img_key: str(curr_image_path), lbl_key: str(curr_mask_path)}
        for curr_image_path, curr_mask_path in zip(image_paths, mask_paths)
    ]

    return image_mask_paths


def _get_paths(folder: Path, pattern: str) -> List[Path]:
    paths = list(folder.glob(pattern=pattern))

    return paths


def create_data_loader(
    dataset: Dataset, batch_size: int = 1, shuffle: bool = True, num_workers: int = 2
) -> DataLoader:
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return data_loader


def get_train_transforms_3d(
    img_key: str, lbl_key: str, spatial_size: Tuple[int, int, int]
) -> Compose:
    train_transforms_3d = Compose(
        [
            CastToTyped(keys=[img_key, lbl_key], dtype=np.float32),
            RandScaleIntensityd(keys=[img_key], factors=(-0.1, 0.1), prob=0.8),
            RandAdjustContrastd(keys=[img_key], gamma=1.5, prob=0.5),
            RandRotate90d(keys=[img_key, lbl_key], prob=0.2),
            RandFlipd(keys=[img_key, lbl_key], prob=0.2, spatial_axis=2),
            RandFlipd(keys=[img_key, lbl_key], prob=0.2, spatial_axis=1),
            RandFlipd(keys=[img_key, lbl_key], prob=0.2, spatial_axis=0),
            RandRotated(
                keys=[img_key, lbl_key],
                mode=['bilinear', 'nearest'],
                range_x=20,
                range_y=20,
                range_z=20,
                prob=0.3,
                keep_size=True,
            ),
            RandZoomd(
                keys=[img_key, lbl_key],
                mode=['trilinear', 'nearest'],
                prob=0.4,
                keep_size=True,
            ),
            # SpatialPadd(keys=[img_key, lbl_key], spatial_size=spatial_size),
        ]
    )

    return train_transforms_3d


def get_val_transforms_3d(
    img_key: str, lbl_key: str, spatial_size: Tuple[int, int, int]
) -> Compose:
    val_transform_3d = Compose(
        [
            CastToTyped(keys=[img_key, lbl_key], dtype=np.float32),
            SpatialPadd(keys=[img_key, lbl_key], spatial_size=spatial_size),
        ]
    )

    return val_transform_3d


def get_load_transforms(
    img_key: str,
    lbl_key: str,
    orig_labels: List[int],
    target_labels: List[int],
    spatial_size: Tuple[int, int, int],
) -> Compose:
    assert len(orig_labels) == len(
        target_labels
    ), f'Labels have different size {len(orig_labels)} != {len(target_labels)}'

    transform_3d_cached = Compose(
        [
            LoadImaged(keys=[img_key, lbl_key], dtype=np.float32),
            AddChanneld(keys=[img_key, lbl_key]),
            ScaleIntensityRanged(
                keys=[img_key],
                a_min=-200.0,
                a_max=2500.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CastToTyped(keys=[img_key, lbl_key], dtype=np.float32),
            MapLabelValued(
                keys=[lbl_key], orig_labels=orig_labels, target_labels=target_labels
            ),
            Resized(keys=[img_key, lbl_key], spatial_size=spatial_size),
        ]
    )

    return transform_3d_cached
