import os
import torch
import torch.nn as nn
import gdal
import numpy as np
from PIL import Image
from TileMaker import TileMaker



if __name__ == "__main__":
    cwd = os.getcwd()
    """
    validationdir = os.path.join(cwd,"validation")
    tiledir = os.path.join(validationdir,"validation_tile_index.txt")
    
    subtiledir = os.path.join(validationdir, "sub_tiles")

    tilemaker = TileMaker(tiledir,validationdir)
    tilemaker.write_naip_tiles_singleband(subtiledir)"""
        
    validationdir = os.path.join(cwd,"validation")
    tiledir = os.path.join(validationdir,"validation_tile_index.txt")
    
    subtiledir = "/home/keyang/Desktop/2021igrss_data/DSIFN/TestSet_NAIP"

    tilemaker = TileMaker(tiledir,validationdir, tilesize=512, target_image_size= 4096)
    tilemaker.write_naip_tiles_rgb(subtiledir, saveformat='jpeg')
   
