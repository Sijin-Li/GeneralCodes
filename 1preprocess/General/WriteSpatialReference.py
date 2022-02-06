from TiffReadClipWrite import readtiff2array, writeTifffile
import glob
import os

'''
根据文件名，读取原始tif空间参考，写入results中
'''

if __name__ == "__main__":
    oripath = "E:/Dataset/202101_withDinghu/temp/Dataset/202201datasetV3/size224/s2/test/dem8bit_crop"
    resultpath = "E:/Dataset/202101_withDinghu/temp/Result/GullyDetecAttUnet/auWP_dV3_t10_1070_dem"
    targetpath = resultpath + "_SpaRef"
    os.makedirs(targetpath, exist_ok=True)

    # 遍历原始文件夹所有文件名
    from_names = glob.glob(os.path.join(resultpath, "*.tif"))
    for i in range(len(from_names)):
        img_name = os.path.basename(from_names[i])

        orifile = os.path.join(oripath, img_name)
        resfile = os.path.join(resultpath, img_name)
        tarfile = os.path.join(targetpath, img_name)

        'para = [band, datatype, geoTrans, geoPro]'
        oridata, oripara = readtiff2array(orifile)
        resdata, _ = readtiff2array(resfile)

        writeTifffile(resdata, oripara, tarfile)