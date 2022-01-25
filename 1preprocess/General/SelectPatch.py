# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 09:33:47 2020

@author: LLLLSJ
"""

# random select patch

from TiffReadClipWrite import readtiff2array
import numpy as np
from osgeo import gdal
import glob
import os, random, shutil
import copy

'''
判断条件 是否符合要求
返回符合条件的像元个数
在主程序中进行判断 如果返回的index个数比要求的高，则认为符合要求，可供选择
'''
def judge(layer, isSlope = False):
    # 设置判定阈值
    index = 0
    a = layer.shape[0]
    b = layer.shape[1]
    for i in range(a):
        for j in range(b):
            if isSlope is True:
                # 坡度图无效值为255
                if layer[i, j, 0] < 255:
                    index = index + 1
            else:
                # 根据mask筛选
                # if layer[i, j, 0] == 255:
                # 根据image筛选
                if layer[i, j, 0] > 0:
                    index = index + 1
    return index

'''
从原始文件夹向目标文件夹移动文件
随机移动
剪切or复制
'''
if __name__ == "__main__":

    # 筛选比例（筛选数量=总数量*rate）
    rate = 1

    # 当index为None时，表示不进行判断
    index = None

    # 原始路径
    cwd_ori = "E:/Dataset/202101_withDinghu/temp/Dataset/202201datasetV3/size224/s1/train/allpatch"
    # 原始文件夹
    orifolder = ["image_crop", "mask_crop","dem8bit_crop", "dem_crop", "hillshade_crop", "slope_crop"]
    oripath_0 = os.path.join(cwd_ori, orifolder[0])
    oripath_1 = os.path.join(cwd_ori, orifolder[1])
    oripath_2 = os.path.join(cwd_ori, orifolder[2])
    oripath_3 = os.path.join(cwd_ori, orifolder[3])
    oripath_4 = os.path.join(cwd_ori, orifolder[4])
    oripath_5 = os.path.join(cwd_ori, orifolder[5])

    # 后缀名
    # postfix = ""
    # finalpath = os.path.join(oripath, postfix)

    # 遍历原始文件夹所有文件名
    pathDir = os.listdir(oripath_0)
    filenumber = len(pathDir)

    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片

    # 目标路径
    cwd_tar = "E:/Dataset/202101_withDinghu/temp/Dataset/202201datasetV3/size224/s1/train/selected"
    # 目标文件夹
    tarfolder = ["image_crop", "mask_crop","dem8bit_crop", "dem_crop", "hillshade_crop", "slope_crop"]
    tarpath_0 = os.path.join(cwd_tar, tarfolder[0])
    tarpath_1 = os.path.join(cwd_tar, tarfolder[1])
    tarpath_2 = os.path.join(cwd_tar, tarfolder[2])
    tarpath_3 = os.path.join(cwd_tar, tarfolder[3])
    tarpath_4 = os.path.join(cwd_tar, tarfolder[4])
    tarpath_5 = os.path.join(cwd_tar, tarfolder[5])

    os.makedirs(tarpath_0, exist_ok=True)
    os.makedirs(tarpath_1, exist_ok=True)
    os.makedirs(tarpath_2, exist_ok=True)
    os.makedirs(tarpath_3, exist_ok=True)
    os.makedirs(tarpath_4, exist_ok=True)
    os.makedirs(tarpath_5, exist_ok=True)

    for img_name in sample:
        orifile_0 = os.path.join(oripath_0, img_name)
        tarfile_0 = os.path.join(tarpath_0, img_name)

        orifile_1 = os.path.join(oripath_1, img_name)
        tarfile_1 = os.path.join(tarpath_1, img_name)
        
        orifile_2 = os.path.join(oripath_2, img_name)
        tarfile_2 = os.path.join(tarpath_2, img_name)

        orifile_3 = os.path.join(oripath_3, img_name)
        tarfile_3 = os.path.join(tarpath_3, img_name)

        orifile_4 = os.path.join(oripath_4, img_name)
        tarfile_4 = os.path.join(tarpath_4, img_name)
                
        orifile_5 = os.path.join(oripath_5, img_name)
        tarfile_5 = os.path.join(tarpath_5, img_name)

        # 判断后移动
        labellayer, _ = readtiff2array(orifile_0)
        index = judge(labellayer)
        if index > 50:
            print(index)
            # 复制
            # shutil.copyfile(orifile_0, tarfile_0)
            # shutil.copyfile(orifile_1, tarfile_1)
            # shutil.copyfile(orifile_2, tarfile_2)
            # shutil.copyfile(orifile_3, tarfile_3)
            shutil.copyfile(orifile_4, tarfile_4)
            # shutil.copyfile(orifile_5, tarfile_5)

        # 复制
        # shutil.copyfile(orifile_0, tarfile_0)
        # shutil.copyfile(orifile_1, tarfile_1)
        # shutil.copyfile(orifile_2, tarfile_2)
        # shutil.copyfile(orifile_3, tarfile_3)
        # shutil.copyfile(orifile_4, tarfile_4)
        # shutil.copyfile(orifile_5, tarfile_5)
        
        # 剪切
        # shutil.move(orifile, tarfile)
        # shutil.move(orifile_2, tarfile_2)
        # shutil.move(orifile_3, tarfile_3)
        # shutil.move(orifile_4, tarfile_4)
        # shutil.move(orifile_5, tarfile_5)
        

    print('done')
