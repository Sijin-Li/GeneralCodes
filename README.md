
# GeneralCodes

## Introduction
for studying; CNN preprocessing
Codes here are programmed by myself and are used to simplify the processing in the daily study.

## Plan
### [Preprocessing](https://github.com/Sijin-Li/GeneralCodes/tree/main/1preprocess)
[samples in jupyter notebook](https://github.com/Sijin-Li/GeneralCodes/blob/main/1preprocess/newTestingTiffProcess.ipynb)
> * reading and saving geotiff [TiffReadClipWrite.py](https://github.com/Sijin-Li/GeneralCodes/blob/main/1preprocess/General/TiffReadClipWrite.py)
> * clipping the complete file to small patches [TiffReadClipWrite.py](https://github.com/Sijin-Li/GeneralCodes/blob/main/1preprocess/General/TiffReadClipWrite.py)
> * selecting specific files according to user requirements and moving them to a new path [SelectPatch.py](https://github.com/Sijin-Li/GeneralCodes/blob/main/1preprocess/General/SelectPatch.py)
> * producing training patches with special data types (such as samples for pix2pix which needs concatenating two individual samples) [ConcatenateSamples.py](https://github.com/Sijin-Li/GeneralCodes/blob/main/1preprocess/General/ConcatenateSamples.py)
> * transfer RGB images to HSV and segment shadows according a threshold in V band. (for the gully detection based on UAV images with a large areas of shadows) [ShadowExtraction.py](https://github.com/Sijin-Li/GeneralCodes/blob/main/1preprocess/General/ShadowExtraction.py)
> * mosaic small patches into complete image through GDAL. [GDALmosaic.py](https://github.com/Sijin-Li/GeneralCodes/blob/main/1preprocess/General/GDALmosaic.py)
> * read spatial reference in original imagery and write it in target imagery. [WriteSpatialReference.py](https://github.com/Sijin-Li/GeneralCodes/blob/main/1preprocess/General/WriteSpatialReference.py)
> * read .pt(h) model file and transfer it to other type of models (.h5/.onnx) [TransforPTtoothermodel.py](https://github.com/Sijin-Li/GeneralCodes/blob/main/1preprocess/General/TransforPTtoothermodel.py


### General CNNs
I used UNet, ResNet, ResUNet, pix2pix before, and I'll put them here in the next few weeks.
> * [Net Set](https://github.com/Sijin-Li/GeneralCodes/tree/main/2generalmodel/NetSet_pytorch)
> > in this folder, you could use several net structure including RESUNET, RESUNET_PLUS_PLUS, UNET, AttU_Net in one train.py file. You could adjust the parameter of "MODELTYPE" in .yaml file to select different net structure.
> * UNet
> > * [UNet-Keras](https://github.com/zhixuhao/unet)
> > * [UNet-pytorch-origin version](https://github.com/milesial/Pytorch-UNet)
> > * [Adjusted UNet_pytorch](https://github.com/Sijin-Li/GeneralCodes/blob/main/2generalmodel/ResUNet/unet_model.py) can be found in ResUNet folder. (in train.py, adjusting the model to UNet in Line 42. The tag that can control the selection of models is in .ymal file.)
> > * [Adjusted UNet_Keras](https://github.com/Sijin-Li/GeneralCodes/tree/main/2generalmodel/UNet_Keras/Adjusted) (will be completed in the next few days)
> > * Tips: using mask01_crop (background: 0, target: 1)

> * ResUNet
> > * [origin pytorch version](https://github.com/rishikksh20/ResUnet)
> > * [adjusted version for gully detection](https://github.com/Sijin-Li/GeneralCodes/tree/main/2generalmodel/ResUNet/ResUNet_AdjustedV1)
> > > (the adjusted version1: add testing code; user can determine the learning rate, step size for the decrease of lr, validation steps, input paths and other parameters in .yaml file.)
> > * Tips: using mask_crop (background: 0, target: 255)
> * pix2pix
> > * [keras verison](https://github.com/Sijin-Li/GeneralCodes/tree/main/2generalmodel/pix2pix_keras_adjusted)
> > * [tensorflow version](https://github.com/Sijin-Li/GeneralCodes/tree/main/2generalmodel/pix2pix_tensorflow_adjusted)
> > > I will focus on this repo in the next few weeks. in the current version, I rewrote the generator structure and used ResUNet (in keras Model) as the generator. The discriminator has not been changed until now.
> * [Attention UNet](https://arxiv.org/pdf/1804.03999.pdf) in [Repository](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/Models.py)

# Useful repositories
[Unet-Segmentation-Pytorch-Nest-of-Unets](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets)
> pytorch
> including UNet, RCNN-UNet,  Attention Unet, RCNN-Attention Unet and Nested UNet in Models.py. 
[UNet Zoon](https://github.com/Andy-zhujunwen/UNET-ZOO)
> in pytorch
[GAN Zoo](https://github.com/hindupuravinash/the-gan-zoo)
> a collection of GAN repositories.
[GAN Zoo Pytorch version](https://github.com/eriklindernoren/PyTorch-GAN)
> All GAN completed by pytorch.
[FFT network](https://github.com/JamieGainer/NN_for_FFT_Autoencoded_MNIST)
> I am not clear about it now.
