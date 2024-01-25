import logging
import sys

import torch
from preregistration.imi_varreg.imi_image_warp import write_image_sitk, tensor_image_to_sitk

#global_debug_level = 3


def initialize_dbg_level(value):
    global global_debug_level
    global_debug_level = value
    logging.basicConfig(format='%(levelname)s-%(asctime)s: %(message)s', level=logging.DEBUG, stream=sys.stdout,
                        datefmt='%d/%m/%Y %I:%M:%S %p')


def debug_check_level(level):
    global global_debug_level
    return level <= global_debug_level


def debug_print(level, *args):
    if debug_check_level(level):
        logging.debug(' '.join(map(str, args)))


def debug_print_field_statistics(level, field, name=""):
    if debug_check_level(level):
        assert len(field.size()) == 4 or len(field.size()) == 5
        field_magnitude = field.pow(2).sum(dim=1)
        debug_print(level, f"{name}: mean={field.abs().mean()} range=[{field.min()} - {field.max()}], "
                    f"magnitude mean={field_magnitude.mean()} range=[{field_magnitude.min()} - {field_magnitude.max()}]"
                    f" total={field_magnitude.sqrt().sum()}")


def debug_write_image_sitk(level, image, filename: str, meta_data=None):
    if debug_check_level(level):
        if isinstance(image, torch.Tensor):
            write_image_sitk(tensor_image_to_sitk(image, meta_data), filename)
        else:
            write_image_sitk(image, filename)
