"""" Modified version of https://github.com/jeffwen/road_building_extraction/blob/master/src/utils/data_utils.py """
from __future__ import print_function, division
from torch.utils.data import Dataset
from skimage import io
import glob
import os
import torch
from torchvision import transforms
import random



class ImageDataset(Dataset):
    """Massachusetts Road and Building dataset"""

    def __init__(self, hp, train=True, test=False, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with image paths
            train_valid_test (string): 'train', 'valid', or 'test'
            root_dir (string): 'mass_roads', 'mass_roads_crop', or 'mass_buildings'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.train = train
        self.test = test
        self.inputtype = hp.inputtype
        self.targettype = hp.targettype
        self.shadowtype = hp.shadowtype
        self.invshadowtype = hp.invshadowtype
        if train:
            self.path = hp.train
        elif test:
            self.path = hp.test
        else:
            self.path = hp.valid
        self.input_list = glob.glob(
            os.path.join(self.path, self.inputtype, "*.tif"), recursive=True
        )
        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        imagepath = self.input_list[idx]
        filename = os.path.basename(imagepath) 
        image = io.imread(imagepath)
        if self.test == False:
            mask = io.imread(imagepath.replace(self.inputtype, self.targettype))
            shadow = io.imread(imagepath.replace(self.inputtype, self.shadowtype))
            invshadow = io.imread(imagepath.replace(self.inputtype, self.invshadowtype))
            sample = {"sat_img": image, "map_img": mask, "shd_img": shadow, "inv_img": invshadow,"filename": filename}
        else:           
            sample = {"sat_img": image, "filename": filename}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensorTarget_fortest(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sat_img, filename = sample["sat_img"], sample["filename"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        return {
            "sat_img": transforms.functional.to_tensor(sat_img),
            "filename": filename,
        }  # unsqueeze for the channel dimension

class ToTensorTarget(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sat_img, map_img, shd_img, inv_img, filename = sample["sat_img"], sample["map_img"], sample["shd_img"], sample["inv_img"], sample["filename"]
        ##
        p1 = random.randint(0, 1)
        p2 = random.randint(0, 1)
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p1),
            transforms.RandomVerticalFlip(p2),
            transforms.ToTensor()
        ])
        # transform_shd = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.ToTensor()
        # ])
        sat_img = transform_train(sat_img)
        map_img = transform_train(map_img)
        # shd_img = transform_shd(shd_img)
        # inv_img = transform_shd(inv_img)
        ##

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        return {
            "sat_img": sat_img, #transforms.functional.to_tensor(sat_img),
            "map_img": map_img, #torch.from_numpy(map_img).unsqueeze(0).float().div(355),
            "shd_img": transforms.functional.to_tensor(shd_img),
            "inv_img": transforms.functional.to_tensor(inv_img),
            "filename": filename,
        }  # unsqueeze for the channel dimension


class NormalizeTarget(transforms.Normalize):
    """Normalize a tensor and also return the target"""

    def __call__(self, sample):
        return {
            "sat_img": transforms.functional.normalize(
                sample["sat_img"], self.mean, self.std
            ),
            "map_img": sample["map_img"],
        }


# https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
