"""Module with datasets"""

import abc
from typing import Any, Dict

from torch.utils.data import Dataset


class BaseDataset(Dataset, abc.ABC):
    def __init__(self):
        self.img_key = 'image'
        self.lbl_key = 'mask'

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError('It is base dataset, use implementation!')

    def __len__(self):
        raise NotImplementedError('It is base dataset, use implementation!')
