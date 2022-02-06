import gdal
import gdalconst
import glob
import os


if __name__ == "__main__":
    classname = 'GUN_dV3_t1_280epochs'
    orifilePath = 'E:/Dataset/202101_withDinghu/temp/Result/mosaic-UNet/result_dV3/'+classname+'/W0_0_0.tif'  # Keras
    #"E:/Dataset/202101_withDinghu/temp/Result/GullyDetecAttUnet/"+classname+"/W0_0_0.tif"  # pytorch

    tempfilePath = orifilePath

    outputPath = 'E:/Dataset/202101_withDinghu/temp/Result/mosaic-UNet/mosaic_result_dV3/'+classname+'/'  # Keras
    #'E:/Dataset/202101_withDinghu/temp/Result/GullyDetecAttUnet/RasterMosaic/'+classname+'/' # pytorch
    os.makedirs(outputPath, exist_ok=True)
    j=0

    inputPath = 'E:/Dataset/202101_withDinghu/temp/Result/mosaic-UNet/result_dV3/'+classname+'/'

    from_names = glob.glob(os.path.join(inputPath, "*.tif"))
    for i in range(len(from_names)):
        # outputfilePath = os.path.join(outputPath, str(j), ".tif")
        outputfilePath = outputPath+str(j)+'.tif'
        j=j+1

        print(os.path.basename(from_names[i]))

        tempfile = gdal.Open(tempfilePath, gdal.GA_ReadOnly)
        tempProj = tempfile.GetProjection()

        inputfilePath = os.path.join(inputPath, os.path.basename(from_names[i]))
        inputrasfile = gdal.Open(inputfilePath, gdal.GA_ReadOnly)
        inputProj = inputrasfile.GetProjection()

        options=gdal.WarpOptions(srcSRS=tempProj, dstSRS=tempProj,format='GTiff',resampleAlg=gdalconst.GRA_Bilinear)
        gdal.Warp(outputfilePath,[tempfile,inputrasfile],options=options)

        tempfilePath = outputfilePath