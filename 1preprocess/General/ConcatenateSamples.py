# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 22:10:05 2020

@author: LLLLSJ
"""
# current for generating basic samples with 3 bands

# concatenate gully line and ridge line
import numpy as np
from osgeo import gdal
import glob
import os
import copy
import imageio
import cv2
from General.TiffReadClipWrite import write_to_disk


def concate_samples(sample1, sample2, savepath, filename):
    sample1 = sample1.reshape((1, sample1.shape[0], sample1.shape[1], sample1.shape[2]))
    sample2 = sample2.reshape((1, sample2.shape[0], sample2.shape[1], sample2.shape[2]))

    sample1band = sample1.shape[3]
    sample2band = sample2.shape[3]
    print(sample1band, sample2band)

    # if two samples bands == 1, then firstly concatenating them and secondly repeating it
    if sample1band == 1 and sample2band == 1:
        temp = np.concatenate((sample1, sample2), axis=2)
        temp = temp.repeat(3, axis=3)
    # if two samples bands == 3, then concatenating them
    elif sample1band == 3 and sample2band == 3:
        temp = np.concatenate((sample1, sample2), axis=2)
    elif sample1band == 3 and sample2band == 1:
        sample2 = sample2.repeat(3, axis=3)
        temp = np.concatenate((sample1, sample2), axis=2)
        print('3')
    elif sample1band == 1 and sample2band == 3:
        sample1 = sample1.repeat(3, axis=3)
        temp = np.concatenate((sample1, sample2), axis=2)
    else:
        print("The situation is not defined.")

    out_data = np.squeeze(temp, axis=0)
    #        out_data = out_data.astype(np.uint8)  # data type
    print(out_data.shape)
    write_to_disk(savepath, out_data, filename)

