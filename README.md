
# GeneralCodes

## Introduction
for studying; CNN preprocessing
Codes here are programmed by myself and are used to simplify the processing in the daily study.

## Plan:
## [Preprocessing](https://github.com/Sijin-Li/GeneralCodes/tree/main/1preprocess)
[samples in jupyter notebook](https://github.com/Sijin-Li/GeneralCodes/blob/main/1preprocess/newTestingTiffProcess.ipynb)
> * reading and saving geotiff [TiffReadClipWrite.py](https://github.com/Sijin-Li/GeneralCodes/blob/main/1preprocess/General/TiffReadClipWrite.py)
> * clipping the complete file to small patches [TiffReadClipWrite.py](https://github.com/Sijin-Li/GeneralCodes/blob/main/1preprocess/General/TiffReadClipWrite.py)
> * selecting specific files according to user requirements and moving them to a new path [SelectPatch.py](https://github.com/Sijin-Li/GeneralCodes/blob/main/1preprocess/General/SelectPatch.py)
> * producing training patches with special data types (such as samples for pix2pix which needs concatenating two individual samples) [ConcatenateSamples.py](https://github.com/Sijin-Li/GeneralCodes/blob/main/1preprocess/General/ConcatenateSamples.py)

## General CNNs
I used UNet, ResNet, ResUNet, pix2pix before, and I'll put them here in the next few weeks.
> * UNet
> * ResUNet
> * pix2pix

