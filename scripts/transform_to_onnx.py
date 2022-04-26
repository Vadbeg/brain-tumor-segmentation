"""Script for converting model to ONNX format."""

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import onnx
import onnxruntime
import torch
import typer
from utils import add_parent_dir_to_sys

add_parent_dir_to_sys()

from brain_tumor_segmentation.modules.model.dense_vnet import DenseVNet  # noqa: E402


def _reformat_checkpoint(
    checkpoint: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    new_checkpoint = OrderedDict()

    for key, item in checkpoint.items():
        if key.startswith('model.'):
            new_checkpoint[key[6:]] = item

    return new_checkpoint


def _load_and_check_onnx_model(onnx_model_path: Path) -> onnx.ModelProto:
    onnx_model = onnx.load(str(onnx_model_path))
    onnx.checker.check_model(onnx_model)

    return onnx_model


def _test_onnx_model(
    onnx_model_path: Path, x: torch.Tensor, torch_out: torch.Tensor
) -> None:
    ort_session = onnxruntime.InferenceSession(str(onnx_model_path))

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def _transform_to_onnx(
    lightning_model_path: Path = typer.Option(
        default=..., help='Path to trained lightning model'
    ),
    onnx_res_path: Path = typer.Option(
        default='res.onnx', help='Path to result onnx model'
    ),
    in_channels: int = typer.Option(default=1, help='Number of input model channels'),
    out_channels: int = typer.Option(default=2, help='Number of output model channels'),
    tensor_size: Tuple[int, int, int, int] = typer.Option(
        default=(1, 184, 184, 128), help='Number of output model channels'
    ),
    opset_version: int = typer.Option(default=11, help='ONNX opset version'),
) -> None:
    """
    Function for transforming Pytorch-Lightning model to ONNX model
    """

    device = torch.device('cpu')

    state_dict = torch.load(lightning_model_path, map_location=device)['state_dict']
    state_dict = _reformat_checkpoint(checkpoint=state_dict)

    model = DenseVNet(in_channels=in_channels, out_channels=out_channels)
    model.load_state_dict(state_dict=state_dict)
    model.eval()

    x = torch.randn(1, *tensor_size, requires_grad=True)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(
        model=model,  # model being run
        args=x,  # model input (or a tuple for multiple inputs)
        f=onnx_res_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=opset_version,  # the ONNX version to export the model to
        do_constant_folding=True,
        # whether to execute constant folding for optimization
        input_names=['input'],  # the model's input names
        output_names=['output'],  # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},  # variable length axes
            'output': {0: 'batch_size'},
        },
    )

    _load_and_check_onnx_model(onnx_model_path=onnx_res_path)
    _test_onnx_model(onnx_model_path=onnx_res_path, x=x, torch_out=torch_out)


if __name__ == '__main__':
    typer.run(_transform_to_onnx)
