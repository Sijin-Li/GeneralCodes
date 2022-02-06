from TiffReadClipWrite import readtiff2array

import os

'''
1. 读取单幅结果+参考真值+test流域范围
2. 判断
2.1 要排除nodata

for循环
    查询watershed有效值范围 做判断
    如果 为有效值像素
        如果ref和result == 1
            则TP++1
        if ref == 1 & result == 0
            FN++
        if ref == 0 & result == 1
            FP++
        if ref == 0 & result == 0
            TN++
'''

if __name__ == "__main__":
    resultfilepath = "E:/Dataset/202101_withDinghu/temp/Result/mosaic-UNet/mosaic_result_dV3/GUN_dV3_t1_280epochs_05.tif"
    # "E:/Dataset/202101_withDinghu/temp/Result/GullyDetecAttUnet/RasterMosaic/lr001/mask/03-04.tif"
    referefilepath = "E:/Dataset/202101_withDinghu/temp/subAreas/V3/forAccuracyAssessment/test/label/Wang_label_frotrain.tif"
    # test area
    watershedfilepath = "E:/Dataset/202101_withDinghu/temp/subAreas/V3/forAccuracyAssessment/test/watershed/wangmaogou_watershed_fortrain.tif"

    result, _ = readtiff2array(resultfilepath)
    refere, _ = readtiff2array(referefilepath)
    watershed, _ = readtiff2array(watershedfilepath)

    w = refere.shape[0]
    h = refere.shape[1]
    i = 0
    j = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(w):
        for j in range(h):
            if watershed[i,j] == 1:
                if refere[i,j] == 1 and result[i,j] == 1:
                    TP = TP + 1
                elif refere[i,j] == 1 and result[i,j] == 0:
                    FN = FN + 1
                elif refere[i,j] == 0 and result[i,j] == 1:
                    FP = FP + 1
                elif refere[i,j] == 0 and result[i,j] == 0:
                    TN = TN + 1
    accuracy = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2*TP / (2*TP+FP+FN)

    print(os.path.basename(resultfilepath))
    print(TP)
    print(TN)
    print(FP)
    print(FN)
    print(accuracy)
    print(recall)
    print(F1)