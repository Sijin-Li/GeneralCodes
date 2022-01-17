# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 13:53:44 2020

@author: LLLLSJ
"""


import numpy as np
from osgeo import gdal
import os
import imageio
import glob

'''
readtiff2array: read tif file and output an numpy array
calculateTransform: calculating 'geotrans' affine transformation 
clipgeotiff_topatch: clip the complete data to small patches. (nonrandom)
'''


def readtiff2array(oriPath):
    in_ds = gdal.Open(oriPath)
    print("open tif file succeed")

    # 读取原图中的每个波段
    row = in_ds.RasterYSize  # 行
    col = in_ds.RasterXSize  # 列
    band = in_ds.RasterCount  # 波段
    geoTrans = in_ds.GetGeoTransform()
    geoPro = in_ds.GetProjection()

    # specific datatype
    #     newDataType = gdal.GDT_UInt16;
    #     data = np.zeros([row, col, band], newDataType)  # 建立数组保存读取的tiff

    # according to the input datatype
    datatype_index = in_ds.GetRasterBand(1).DataType
    datatype = gdal.GetDataTypeName(datatype_index)
    if 'GDT_Byte' in datatype:
        newDataType = 'int8'
    elif 'GDT_UInt16' in datatype:
        newDataType = 'int16'
    else:
        newDataType = 'float32'
    data = np.zeros([row, col, band], newDataType)  # 建立数组保存读取的tiff

    for i in range(band):
        dt = in_ds.GetRasterBand(i + 1)
        # 从每个波段中裁剪需要的矩形框内的数据
        data[:, :, i] = dt.ReadAsArray(0, 0, col, row)

    del in_ds

    return data, [band, datatype, geoTrans, geoPro]
    # data shape = HWC


'''
16bit to 8bit
for slope data, nodata value = 91, therefore we need to find a maximum except 91 and view it as the max_16bit
'''


def transfer_16bit_to_8bit(data, isSlope, nodatavalue=91):
    # transform to 8bit
    min_16bit = 0
    max_16bit = 0
    if isSlope is True:

        # 先对有效值区域进行处理，找到有效值的最大最小值
        ## 找到有效值的所有索引
        indexLS = np.argwhere(data < nodatavalue)
        ## indexLS为tuple，shape=[n,3],n代表有多少行
        ## 具体在图像中的坐标为 indexLS.shape[0][0], indexLS.shape[0][1]
        rows = indexLS.shape[0]
        i = 0
        for i in range(rows):
            ## 找到其临时值
            tempValue = data[indexLS[i][0], indexLS[i][1], 0]
            ## 对临时值进行判断
            if tempValue != 91:
                if tempValue > max_16bit:
                    max_16bit = tempValue
                elif tempValue < min_16bit:
                    min_16bit = tempValue

                    # 进而对无效值（通常为91）进行处理，思路为将其改为最大值+最大值的10%
        # 循环过程与上一步类似
        indexGE = np.argwhere(data >= nodatavalue)
        rows = indexGE.shape[0]
        i = 0
        for i in range(rows):
            data[indexGE[i][0], indexGE[i][1], 0] = max_16bit  # + max_16bit*0.1
        print(max_16bit, min_16bit)
    else:
        min_16bit = np.min(data)
        max_16bit = np.max(data)

    data_8bit = np.array(np.rint(255 * ((data - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
    return data_8bit


'''
仿射变换
'''


def calculateTransform(ori_transform, offsetX, offsetY):
    # 读取原图仿射变换参数值
    top_left_x = ori_transform[0]  # 左上角x坐标
    w_e_pixel_resolution = ori_transform[1]  # 东西方向像素分辨率
    top_left_y = ori_transform[3]  # 左上角y坐标
    n_s_pixel_resolution = ori_transform[5]  # 南北方向像素分辨率

    # 根据反射变换参数计算新图的原点坐标
    top_left_x = top_left_x + offsetX * w_e_pixel_resolution
    top_left_y = top_left_y + offsetY * n_s_pixel_resolution

    # 将计算后的值组装为一个元组，以方便设置
    dst_transform = (top_left_x, ori_transform[1], ori_transform[2], top_left_y, ori_transform[4], ori_transform[5])

    return dst_transform
'''
裁剪主函数 调用topatch
typeName & index: 输入数据编号
'''
def clipgeotiff(inputPath, savePath, patchSize, patchIntersection, startCol, startRow, typeName, index, To8bit=False, isSlope=False):
    os.makedirs(savePath, exist_ok=True)
    # 遍历文件夹
    from_names = glob.glob(os.path.join(inputPath, "*.tif"))

    # 左上角坐标移动步长
    stride = patchSize - int((patchSize - patchIntersection) / 2)

    for i in range(len(from_names)):
        print(from_names[i])
        filePath = os.path.join(inputPath, os.path.basename(from_names[i]))
        # oriPara=[band, datatype, geoTrans, geoPro]
        inputData, oriPara = readtiff2array(filePath)

        # 16bit to 8bit
        if To8bit is True:
            inputData = transfer_16bit_to_8bit(inputData, isSlope)

        # 裁剪 起始坐标
        # 原始代码的[offsetY, offsetX] = [offsetRow, offsetCol]
        offsetCor = [startRow, startCol]
        while offsetCor[1] < inputData.shape[1] - patchSize:
            while offsetCor[0] < inputData.shape[0] - patchSize:
                filename = typeName + str(index) + '_' + str(offsetCor[1]) + '_' + str(offsetCor[0]) + '.tif'
                saveName = os.path.join(savePath, filename)
                cliptopatch(inputData, oriPara[2], oriPara[3], saveName, offsetCor, patchSize)
                offsetCor[0] = offsetCor[0] + stride
            offsetCor[1] = offsetCor[1] + stride
            offsetCor[0] = 0
        index = index + 1

'''
裁剪至小块 并保存
inputData: array
offsetCor: patch的左上角位置 原始代码的[offsetY, offsetX] = [offsetRow, offsetCol]
saveName
patchSize
'''
def cliptopatch(inputData, ori_geoTrans, ori_geoPro, saveName, offsetCor, patchSize):
    # 读取输入数据信息
    ori_Datatype = inputData.dtype
    # inputData
    if len(inputData.shape) == 3:
        in_bands = inputData.shape[2]
    else:
        in_bands = 1

    # 读取要裁剪的原图
    out_band = np.zeros([patchSize, patchSize, in_bands], ori_Datatype)
    # print(out_band.shape)
    for i in range(in_bands):
        out_band[:, :, i] = inputData[offsetCor[0]:offsetCor[0] + patchSize, offsetCor[1]:offsetCor[1] + patchSize, i]

    # 获取原图的原点坐标信息
    ori_transform = ori_geoTrans
    # 计算仿射变化参数
    dst_transform = calculateTransform(ori_transform, offsetCor[1], offsetCor[0])

    # 设置DataType
    if 'int8' in out_band.dtype.name:
        newDataType = gdal.GDT_Byte
    elif 'int16' in out_band.dtype.name:
        newDataType = gdal.GDT_UInt16
    else:
        newDataType = gdal.GDT_Float32

    # 创建gtiff 并 写入
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(saveName, patchSize, patchSize, in_bands, newDataType)
    if (dataset != None):
        dataset.SetGeoTransform(dst_transform)  # 写入仿射变换参数
        dataset.SetProjection(ori_geoPro)  # 写入投影
    for i in range(in_bands):
        dataset.GetRasterBand(i + 1).WriteArray(out_band[:, :, i])
    del dataset

    print("End!")
'''
单独保存函数
gdal 保存为tif
'''
def writeTifffile(out_band, saveName):
    # 设置DataType
    if 'int8' in out_band.dtype.name:
        newDataType = gdal.GDT_Byte
    elif 'int16' in out_band.dtype.name:
        newDataType = gdal.GDT_UInt16
    else:
        newDataType = gdal.GDT_Float32

    # 读取im_data形状
    if len(out_band.shape) == 3:
        im_height, im_width, im_bands = out_band.shape
        # print('1',out_band.shape)
    elif len(out_band.shape) == 2:
        out_band = np.array([out_band])
        im_bands = 1
        # print('2',out_band.shape)
    else:
        im_bands, (im_height, im_width) = 1, out_band.shape
        # print('3')

    # 创建gtiff 并 写入
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(saveName, patchSize, patchSize, oriPara(0), newDataType)
    if (dataset != None):
        dataset.SetGeoTransform(dst_transform)  # 写入仿射变换参数
        dataset.SetProjection(oriPara(3))  # 写入投影
    for i in range(oriPara(0)):
        dataset.GetRasterBand(i + 1).WriteArray(out_band[:, :, i])
    del dataset

    print("End!")

'''
保存为PNG
'''
def write_to_disk(savePath, img_data, from_names, over_write=False):

    os.makedirs(savePath, exist_ok=True)

    full_name = from_names.replace(".tif", ".png")
    print(img_data.dtype)
    save_filePath = os.path.join(savePath, full_name)
    
#    im = Image.fromarray(np.uint16(img_data))
#    im.save(save_filePath) #--> 16bit
    
    print(save_filePath)
    imageio.imwrite(save_filePath,img_data, 'PNG-FI') #(save_filePath, img_data)  #imageio.imwrite
    print("Done!")


