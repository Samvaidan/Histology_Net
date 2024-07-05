import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
from glob import glob
import torchvision
import torchvision.transforms as T
from PIL import Image
import time

import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_zenodo import Zenodo_dataset
from datasets.dataset_histology import Histology, HistologyValidation
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from os.path import basename 
from monai.networks.nets.unet import UNet
from monai.metrics import DiceMetric
import torch.onnx
from monai.transforms import SaveImage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
parser.add_argument('--is_savenii', default=True, action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

def inference(image, img_path, net):
    image = image.squeeze(0).cpu().detach().numpy()
    input = torch.from_numpy(image).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        prediction, _, _, _, _ = net(input)
        prediction = prediction.squeeze(0)
    return prediction


def dice_score(pred, target, smooth=1e-6):
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    
    dice = (2. * intersection + smooth) / (union + smooth)

    return dice.item()
    

def evaluate_performance(model: torch.nn.Module, image):
    image = image.to(torch.device('cuda'))
    with torch.no_grad():
        confidence_maps = model(image)
        confidence_maps = confidence_maps.squeeze(1)
    predicted_mask = confidence_maps > 0.5
    return predicted_mask


if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

dataset_config = {
    'Synapse': {
        'Dataset': Zenodo_dataset, #Synapse_dataset,
        'volume_path': '../data/Synapse/test_vol_h5',
        'list_dir': './lists/lists_Synapse',
        'num_classes': 1,
        'z_spacing': 1,
    },
}
dataset_name = args.dataset
args.num_classes = dataset_config[dataset_name]['num_classes']
args.volume_path = dataset_config[dataset_name]['volume_path']
args.Dataset = dataset_config[dataset_name]['Dataset']
args.list_dir = dataset_config[dataset_name]['list_dir']
args.z_spacing = dataset_config[dataset_name]['z_spacing']
args.is_pretrain = True

# name the same snapshot defined in train script!
args.exp = 'TU_' + dataset_name + str(args.img_size)
snapshot_path = ""


config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.n_classes = args.num_classes
config_vit.n_skip = args.n_skip
config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
if args.vit_name.find('R50') !=-1:
    config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

snapshot = os.path.join('./saved_model/best_model.pth')
net.load_state_dict(torch.load(snapshot))

model_unet = UNet(
    spatial_dims=2,
    in_channels=3, 
    out_channels=1, 
    channels=(16, 32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2, 2)
)

model_unet_2 = UNet(
    spatial_dims=2,
    in_channels=3, 
    out_channels=1, 
    channels=(16, 32, 64, 128, 256, 512, 1024),
    strides=(2, 2, 2, 2, 2, 2),
    num_res_units = 2
)

state_dict = torch.load("./saved_model/best_unet1.pth", map_location='cuda')

model_unet.load_state_dict(state_dict)
model_unet.to(device)
model_unet.eval()

state_dict_2 = torch.load("./saved_model/best_unet2.pth", map_location='cuda')

model_unet_2.load_state_dict(state_dict_2)
model_unet_2.to(device)
model_unet_2.eval()


db_test = Histology("./data/Final_test") 
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

# test_dir = '/home/arnav/Disk/HistologyNet/TransNuSS/transformed_data/validation/labelled'
# val_loader = DataLoader(HistologyValidation(test_dir), batch_size=1, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# with torch.no_grad():
#     dice = []
#     tim = []
#     for val_data in val_loader:
#         val_image, val_label, img_path = val_data[0].to(device), val_data[1].to(device), val_data[2]
#         # val_outputs = model(val_images)
#         st = time.time()

#         output_image_1 = inference(val_image, img_path, net)
#         output_image_2 = torch.round(torch.sigmoid(evaluate_performance(model_unet, val_image)))
#         output_image_3 = torch.round(torch.sigmoid(evaluate_performance(model_unet_2, val_image)))
#         end = time.time()-st
#         tim.append(end)
#         transform = T.ToPILImage()
#         final_output = output_image_1 * 1 + output_image_2 * 0 + output_image_3 * 0
#         final_output[final_output>0.5] = 1
#         final_output[final_output<1] = 0
#         dice.append(dice_score(final_output, val_label.squeeze(0)))
       
#     print("Dice :", np.array(dice).mean(), "Inference Time: ", np.array(tim).mean())

        
for image, img_path in testloader:
    output_image_1 = inference(image, img_path, net)
    output_image_2 = torch.round(torch.sigmoid(evaluate_performance(model_unet, image)))
    output_image_3 = torch.round(torch.sigmoid(evaluate_performance(model_unet_2, image)))

    final_output = output_image_1 * 0.5 + output_image_2 * 0.1 + output_image_3 * 0.4
    final_output[final_output>0.5] = 1
    final_output[final_output<1] = 0 

    # print("The final ouput shape is ", final_output.shape)

    transform = T.ToPILImage()
    final_image = transform(final_output)

    img_path = basename(img_path[0])
    final_image.save(f'./test_images/{img_path}')

    print("The images are created, check'em out girrlll")

