import os
import torch
import torch.nn as nn
import gdal
import numpy as np
from PIL import Image
from TileMaker import TileMaker

if __name__ == "__main__":
    cwd = os.getcwd()

    validationdir = os.path.join(cwd, "validation")
    tiledir = os.path.join(validationdir, "validation_tile_index.txt")
    subtiledir = os.path.join(validationdir, "sub_tiles")

    tilemaker = TileMaker(tiledir, validationdir)
    tilemaker.write_naip_tiles(subtiledir)