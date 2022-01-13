# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 09:33:47 2020

@author: LLLLSJ
"""

# random select patch

from AboutTiffFile import readtiff2array, write_to_disk
import numpy as np
import gdal
import glob
import os, random, shutil
import copy
'''
从原始文件夹向目标文件夹移动文件
随机移动
剪切or复制
'''
if __name__ == "__main__":

    # 筛选比例（筛选数量=总数量*rate）
    rate = 1

    # 原始路径
    cwd_ori = "E:/data/@forLandformClassification/OldPC/data_0710/"
    # 原始文件夹
    orifolder = ["test_b", "test_e", "test_y", "mask"]
    oripath = os.path.join(cwd_ori, orifolder[0])
    oripath_2 = os.path.join(cwd_ori, orifolder[1])
    oripath_3 = os.path.join(cwd_ori, orifolder[2])
    # oripath_4 = os.path.join(cwd_ori, orifolder[3])
    # oripath_5 = os.path.join(cwd_ori, orifolder[4])

    # 后缀名
    # postfix = ""
    # finalpath = os.path.join(oripath, postfix)

    # 遍历原始文件夹所有文件名
    pathDir = os.listdir(oripath)
    filenumber = len(pathDir)

    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片

    # 目标路径
    cwd_tar = "E:/data/@forLandformClassification/v2/valid/"
    # 目标文件夹
    tarfolder = ["input_crop", "DEM", "slope", "mask_crop"]
    tarpath = os.path.join(cwd_tar, tarfolder[0])
    # tarpath_2 = os.path.join(cwd_tar, tarfolder[1])
    # tarpath_3 = os.path.join(cwd_tar, tarfolder[2])
    tarpath_4 = os.path.join(cwd_tar, tarfolder[3])
    #   tarpath_5 = os.path.join(cwd_tar, tarfolder[4])

    for img_name in sample:
        orifile = os.path.join(oripath, img_name)
        tarfile = os.path.join(tarpath, img_name)
        
        # orifile_2 = os.path.join(oripath_2,postfix, img_name)
        # tarfile_2 = os.path.join(tarpath, img_name)
        #
        # orifile_3 = os.path.join(oripath_3,postfix, img_name)
        # tarfile_3 = os.path.join(tarpath, img_name)
        #
        orifile_4 = os.path.join(oripath_4, img_name)
        tarfile_4 = os.path.join(tarpath_4, img_name)
                
        # orifile_5 = os.path.join(oripath_5, img_name)
        # tarfile_5 = os.path.join(tarpath_5, img_name)
        
        # 复制
        # shutil.copyfile(orifile, tarfile)
        # shutil.copyfile(orifile_2, tarfile_2)
        # shutil.copyfile(orifile_3, tarfile_3)
        # shutil.copyfile(orifile_4, tarfile_4)
       # shutil.copyfile(orifile_5, tarfile_5)
        
        # 剪切
        shutil.move(orifile, tarfile)
        # shutil.move(orifile_2, tarfile_2)
        # shutil.move(orifile_3, tarfile_3)
        shutil.move(orifile_4, tarfile_4)
        # shutil.move(orifile_5, tarfile_5)
        

    print('done')
