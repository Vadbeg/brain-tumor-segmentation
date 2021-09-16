"""Module with train segmentation dataset"""

from typing import Dict, List, Optional

import numpy as np
from monai.transforms import Compose

from brain_tumor_segmentation.modules.data.datasets.base_dataset import BaseDataset


class BrainSegDataset(BaseDataset):
    def __init__(
        self,
        image_mask_paths: List[Dict[str, str]],
        load_transforms: Compose,
        aug_transforms: Optional[Compose] = None,
    ):
        super().__init__()

        self.image_mask_paths = image_mask_paths

        self.load_transforms = load_transforms
        self.aug_transforms = aug_transforms

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        curr_image_mask_path = self.image_mask_paths[idx]
        image_and_mask = self.load_transforms(curr_image_mask_path)

        if self.aug_transforms:
            image_and_mask = self.aug_transforms(image_and_mask)

        image_and_mask[self.lbl_key] = np.int8(image_and_mask[self.lbl_key] != 0)

        return image_and_mask

    def __len__(self) -> int:
        return len(self.image_mask_paths)
