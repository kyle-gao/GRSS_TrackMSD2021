import urllib.request
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import shutil
import gdal
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torchvision
from PIL import Image



class TileMaker():

    def __init__(self, tile_index, root_dir, tilesize=390, image_size=3880):
        """
        args:
        str-tile_index:str list of naip tile indexes
        str-root_dir: root directory
        tilesize = size of sub tiles we wish to create
        image_size = size of naip images
        """
        with open(tile_index) as f:
            tiles = f.readlines()
            tiles = [t.replace("\n", "") for t in tiles]
            self.tiles = tiles
            self.root_dir = root_dir
            self.tilesize = tilesize
            self.image_size = image_size
            self.__repeats = int(np.ceil(image_size / tilesize))

    def __make_tiles(self, image):
        """
        Args:
        torch array - image (bands,w,h)

        splits an image into (n*n) tiles where n-repeats
        image must be (size*n,size*n)
        output:
        torch array - (ntiles, bands, w, h)
        """
        image = image.unsqueeze(0)
        splits = [self.tilesize for _ in range(self.__repeats)]
        tiles = image.split(splits, dim=-1)
        tiles = torch.cat(tiles, axis=0)
        tiles = tiles.split(splits, dim=-2)
        tiles = torch.cat(tiles)
        return tiles

    def __get_tensor(self, imagename, pad=True, unsqueeze=False):
        """args
        -imagename pathname for a tiff file
        returns a float32 torch tensor

        """

        image = torch.tensor(gdal.Open(imagename).ReadAsArray())
        if pad:
            image = F.pad(image, [10, 10, 10, 10])
        if unsqueeze:
            image = image.unsqueeze(0)
        return image

    def write_naip_tiles(self, tiledir, return_tiles_and_bands=False):
        """
        writes tiles to target directory
        Args
        str - tiledir : target directory

        """

        if not (os.path.isdir(tiledir)):
            os.mkdir(tiledir)

        supertiles = self.tiles
        # (naip images in the datasets are also called tiles, we are making tiles out of these tiles)
        for supertile in supertiles:

            img_name2013 = os.path.join(self.root_dir, supertile + "_naip-2013.tif")
            img_name2017 = os.path.join(self.root_dir, supertile + "_naip-2017.tif")

            image2013 = self.__get_tensor(img_name2013)
            image2017 = self.__get_tensor(img_name2017)

            batches2013 = self.__make_tiles(image2013)
            batches2017 = self.__make_tiles(image2017)

            # (25,nbands,780,780)
            ntiles, bands, _, _ = batches2013.shape

            for tile in range(ntiles):
                for band in range(bands):
                    # tilename format /content/tiles/2002_99_0_naip2013.pt
                    # use tilename.split("_") = ['/content/tiles/2002', '99', '0', 'naip2013.pt'] to reacquire tile and band
                    tilename1 = os.path.join(tiledir, supertile + "_" + str(tile) + "_" + str(band) + "_naip2013.tif")
                    tilename2 = os.path.join(tiledir, supertile + "_" + str(tile) + "_" + str(band) + "_naip2017.tif")
                    image1 = Image.fromarray(batches2013[tile, band, :, :].numpy())
                    image2 = Image.fromarray(batches2017[tile, band, :, :].numpy())

                    image1.save(tilename1, format="tiff")
                    image2.save(tilename2, fotmat="tiff")

            if return_tiles_and_bands:
                return ntiles, bands
