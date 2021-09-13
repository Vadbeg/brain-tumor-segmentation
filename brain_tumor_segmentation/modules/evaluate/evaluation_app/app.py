"""App for CT evaluate using DeepGrow model"""

from typing import Tuple

import numpy as np
import scipy.ndimage
from cv2 import cv2

from brain_tumor_segmentation.modules.evaluate.model_evaluator import ModelEvaluator
from brain_tumor_segmentation.modules.train.training import BrainSegmentation3DModel


class EvaluationApp:
    def __init__(
        self,
        model: BrainSegmentation3DModel,
        ct_image: np.ndarray,
        image_size: Tuple[int, int, int] = (224, 224, 224),
    ) -> None:
        self.model = model
        self.model_evaluator = ModelEvaluator(model=model, device=model.device)

        self.ct_image = self._resize_ct(ct=ct_image, ct_size=image_size)
        self.ct_mask = np.zeros_like(self.ct_image)

        self.image_size = image_size
        self.drawing_window_size = image_size[:2]
        self.curr_ct_slice_idx: int = 0

    def start(self) -> None:
        cv2.namedWindow('Evaluation', flags=cv2.WINDOW_NORMAL)
        self._compute_mask()

        while True:
            self._draw_current_slice_and_mask()

            key_idx = cv2.waitKey(33)

            self._handle_slice_idx(key_idx=key_idx)

            if self._pressed_esc(key_idx=key_idx):
                break

    def _draw_current_slice_and_mask(
        self, min_value: int = -200, max_value: int = 2500
    ) -> None:
        ct_slice = self._get_current_ct_slice()
        ct_slice = self._get_current_drawing_slice(
            ct_slice=ct_slice, min_value=min_value, max_value=max_value
        )

        mask_slice = self._get_current_mask_slice()
        mask_slice = self._mask_postprocessing(mask=mask_slice)

        result = cv2.addWeighted(ct_slice, 0.5, mask_slice, 0.5, 0.0)

        cv2.imshow('Evaluation', result)

    def _compute_mask(self):
        self.ct_mask = self.model_evaluator.evaluate(image=self.ct_image)

    def _get_current_ct_slice(self) -> np.ndarray:
        curr_ct_slice = self.ct_image[:, :, self.curr_ct_slice_idx]

        return curr_ct_slice

    def _get_current_mask_slice(self) -> np.ndarray:
        curr_mask_slice = self.ct_mask[:, :, :, self.curr_ct_slice_idx]

        return curr_mask_slice

    def _get_current_drawing_slice(
        self, ct_slice: np.ndarray, min_value: int = -200, max_value: int = 200
    ) -> np.ndarray:
        ct_image_copy = self._convert_image_for_drawing(
            image=ct_slice, min_value=min_value, max_value=max_value
        )

        return ct_image_copy

    def _handle_slice_idx(self, key_idx: int) -> None:
        if key_idx == 43 and self.curr_ct_slice_idx < self.ct_image.shape[2] - 1:
            self.curr_ct_slice_idx += 1

        if key_idx == 45 and self.curr_ct_slice_idx > 0:
            self.curr_ct_slice_idx -= 1

    @staticmethod
    def _pressed_esc(key_idx: int) -> bool:
        if key_idx == 27:
            return True

        return False

    @staticmethod
    def _pressed_restart(key_idx: int) -> bool:
        if key_idx == 114:
            return True

        return False

    @staticmethod
    def _mask_postprocessing(mask: np.ndarray, edge: float = 0.5) -> np.ndarray:
        mask = np.uint8(mask > edge)  # type: ignore
        mask = mask[..., np.newaxis]
        mask = np.uint8(mask * 255)  # type: ignore

        mask = np.concatenate([mask[1], mask[2], mask[3]], axis=-1)

        return mask

    @staticmethod
    def _convert_image_for_drawing(
        image: np.ndarray, min_value: int = -200, max_value: int = 200
    ) -> np.ndarray:
        image_copy = np.copy(image)
        image_copy = image_copy[..., np.newaxis]
        image_copy = np.float32(image_copy)
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2RGB)

        image_copy = np.clip(image_copy, min_value, max_value)
        image_copy = (image_copy - np.min(image_copy)) / (
            np.max(image_copy) - np.min(image_copy)
        )
        image_copy = np.uint8(image_copy * 255)

        return image_copy

    @staticmethod
    def _resize_image(
        image: np.ndarray, image_size: Tuple[int, int] = (512, 512)
    ) -> np.ndarray:
        image = cv2.resize(src=image, dsize=image_size)

        return image

    @staticmethod
    def _resize_ct(
        ct: np.ndarray, ct_size: Tuple[int, int, int] = (512, 512, 256)
    ) -> np.ndarray:
        zoom_factor = [
            first_value / second_value
            for first_value, second_value in zip(ct_size, ct.shape)
        ]
        res_ct = scipy.ndimage.zoom(ct, zoom=zoom_factor)

        assert res_ct.shape == ct_size, f'Bad result size: {res_ct.shape}!={ct_size}'

        return res_ct
