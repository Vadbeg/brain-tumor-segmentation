"""Module with model evaluator"""

import numpy as np
import torch

from brain_tumor_segmentation.modules.train.training import BrainSegmentation3DModel


class ModelEvaluator:
    def __init__(
        self,
        model: BrainSegmentation3DModel,
        min_value: int = -200,
        max_value: int = 200,
        device: torch.device = torch.device('cpu'),
    ):
        self.model = model
        self.model.eval()
        self.model.to(device)

        self.min_value = min_value
        self.max_value = max_value

        self.device = device

    def evaluate(self, image: np.ndarray) -> np.ndarray:
        print(image.max())
        print(image.min())

        image_tensor = self.preprocess_image(
            image=image,
            min_value=self.min_value,
            max_value=self.max_value,
        )
        image_tensor = image_tensor.to(self.device)

        mask = self.model.forward(image=image_tensor)

        mask = mask.cpu().detach().numpy()
        mask = mask[0]

        return mask

    @staticmethod
    def preprocess_image(
        image: np.ndarray,
        min_value: int = -200,
        max_value: int = 200,
    ) -> torch.Tensor:
        image = np.clip(image, min_value, max_value)
        image = (image - min_value) / (max_value - min_value)

        image_tensor = torch.tensor(image)
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor
