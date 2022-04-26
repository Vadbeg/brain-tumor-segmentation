"""Module with utils for tools module"""

import inspect
import os
import sys


def add_parent_dir_to_sys() -> None:
    current_frame = inspect.currentframe()
    if current_frame is None:
        raise ValueError()
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(current_frame)))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
