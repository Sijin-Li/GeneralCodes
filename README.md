
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

### General CNNs
I used UNet, ResNet, ResUNet, pix2pix before, and I'll put them here in the next few weeks.
> * UNet
> > * [UNet-Keras](https://github.com/zhixuhao/unet)
> > * [UNet-pytorch-origin version](https://github.com/milesial/Pytorch-UNet)
> > * [Adjusted UNet_pytorch](https://github.com/Sijin-Li/GeneralCodes/blob/main/2generalmodel/ResUNet/unet_model.py) can be found in ResUNet folder. (in train.py, adjusting the model to UNet in Line 42. The tag that can control the selection of models is in .ymal file.)
> > * [Adjusted UNet_Keras](https://github.com/Sijin-Li/GeneralCodes/tree/main/2generalmodel/UNet_Keras/Adjusted) (will be completed in the next few days)
> > * Tips: using mask01_crop (background: 0, target: 1)

> * ResUNet
> > * [origin pytorch version](https://github.com/rishikksh20/ResUnet)
> > * [adjusted version for gully detection](https://github.com/Sijin-Li/GeneralCodes/tree/main/2generalmodel/ResUNet/Adjusted)
> > * Tips: using mask_crop (background: 0, target: 255)
> * pix2pix

# Useful repositories
[Unet-Segmentation-Pytorch-Nest-of-Unets](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets)
> pytorch
> including UNet, RCNN-UNet,  Attention Unet, RCNN-Attention Unet and Nested UNet in Modes.py. 

