import warnings

warnings.simplefilter("ignore", (UserWarning, FutureWarning))
from hparams import HParam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import dataloader
import metrics
from models import ResUnet, ResUnetPlusPlus, UNet, AttU_Net
from logger import MyWriter
import torch
import argparse
import os
import numpy as np
from PIL import Image
from tifffile import imsave

'''
out
    resume: checkpoint path
    name: project name

.yaml file
    test path: input file
    save path: saving test result
    RESNET_PLUS_PLUS: False(resunet or resunet++)
'''

def main(hp, resume, name):

    save_dir = "{}/{}".format(hp.save_path, name)
    os.makedirs(save_dir, exist_ok=True)

    # get model
    if hp.MODELTYPE == 'RESUNET':
        model = ResUnet(3).cuda()
    elif hp.MODELTYPE == 'RESUNET_PLUS_PLUS':
        model = ResUnetPlusPlus(3).cuda()
    elif hp.MODELTYPE == 'AttU_Net':
        model = AttU_Net(3,1).cuda()
    else:
        model = UNet(3,1).cuda()

    # loading model from checkpoints
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)

            start_epoch = checkpoint["epoch"]

            best_loss = checkpoint["best_loss"]
            model.load_state_dict(checkpoint["state_dict"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

        # loading test input files
        mass_dataset_test = dataloader.ImageDataset(
            hp, train=False, test=True, transform=transforms.Compose([dataloader.ToTensorTarget_fortest()])
        )
        test_dataloader = DataLoader(
            mass_dataset_test, batch_size=hp.batch_size, num_workers=2, shuffle=False
        )

        model.eval()
        with torch.no_grad():
            for idx, data in enumerate(test_dataloader):

                # get the inputs and wrap in Variable
                inputs = data["sat_img"].cuda()

                outputs = model(inputs)

                output_result = outputs.cpu().data.numpy().astype(np.float32)
                # output_result.shape = (9,1,224,224)
                # print(output_result.shape)
                length = output_result.shape[0]
                i = 0
                for i in range(length):
                  temp = output_result[i,:,:,:] # (1,224,224)        
                  temp = temp.transpose(1,2,0)         
                  # temp = np.reshape(temp,[temp.shape[1],temp.shape[2],temp.shape[0]])
                  # print(temp.shape)
                  # print(data["filename"][i])
                  imsave("{}/{}/{}".format(hp.save_path, name, data["filename"][i]),temp)
                  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road and Building Extraction")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="yaml file for configuration"
    )

    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument("--name", default="default", type=str, help="Experiment name")

    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, "r") as f:
        hp_str = "".join(f.readlines())

    main(hp, resume=args.resume, name=args.name)
    print("done")