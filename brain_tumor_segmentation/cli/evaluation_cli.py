"""Module for evaluation cli"""

import argparse

import nibabel as nib
import numpy as np
import torch
from nibabel.nifti1 import Nifti1Image

from brain_tumor_segmentation.modules.evaluate.evaluation_app.app import EvaluationApp
from brain_tumor_segmentation.modules.train.training import BrainSegmentation3DModel

TUTORIAL = """
It is and CLI for evaluation app

To scroll threw CT press either `+` or `-`
To exit app press `Esc`

You can change size of the app window
"""


def load_ct(ct_path: str) -> np.ndarray:
    ct_all_info: Nifti1Image = nib.load(filename=ct_path)
    orig_ornt = nib.io_orientation(ct_all_info.affine)
    targ_ornt = nib.orientations.axcodes2ornt(axcodes='LPS')
    transform = nib.orientations.ornt_transform(
        start_ornt=orig_ornt, end_ornt=targ_ornt
    )

    img_ornt = ct_all_info.as_reoriented(ornt=transform)

    return img_ornt.get_fdata(dtype=np.float64)


def get_args():
    parser = argparse.ArgumentParser(
        description='Starts simple app for DeepGrow evaluation'
    )

    parser.add_argument(
        '--checkpoint-path', type=str, help='Path to pytorch_lightning checkpoint'
    )
    parser.add_argument(
        '--ct-path', type=str, help='Path to ct file. With *.nii.gz extension'
    )
    parser.add_argument(
        '--device-type',
        type=str,
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Which device type use for inference',
    )

    return parser.parse_args()


def start_eval_app():
    cli_args = get_args()
    print(TUTORIAL)

    checkpoint_path = cli_args.checkpoint_path
    ct_path = cli_args.ct_path
    device_type = cli_args.device_type

    model = BrainSegmentation3DModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path
    )
    model.eval()
    model.to(torch.device(device_type))

    ct_image = load_ct(ct_path=ct_path)

    app = EvaluationApp(model=model, ct_image=ct_image)
    app.start()
