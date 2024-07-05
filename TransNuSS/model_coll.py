from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from monai.networks.nets.unet import UNet
from networks.vit_seg_modeling import VisionTransformer as ViT
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from datasets.dataset_zenodo import Zenodo_dataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.autograd import Function

config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.n_classes = 2
config_vit.n_skip = 3
config_vit.patches.size = (16,16)
config_vit.patches.grid = (int(16), int(16))
transnuss = ViT(config_vit, 256, 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
db_train_lab = Zenodo_dataset(data_path='/home/arnav/Disk/HistologyNet/TransNuSS/data', split="train")
print("The length of training dataset is: {}".format(len(db_train_lab)))

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

class Weighted_Cross_Entropy_Loss(nn.Module):
    """Cross entropy loss that uses weight maps."""

    def __init__(self):
        super(Weighted_Cross_Entropy_Loss, self).__init__()

    def forward(self, pred, target):
        n, c, H, W = pred.shape
        # # Calculate log probabilities
        logp = F.log_softmax(pred, dim=1)

        # Gather log probabilities with respect to target
        logp = torch.gather(logp, 1, target.view(n, 1, H, W))

        # Multiply with weights
        # weighted_logp = (logp * weights).view(n, -1)

        # Rescale so that loss is in approx. same interval
        # weighted_loss = weighted_logp.sum(1) / weights.view(n, -1).sum(1)

        # Average over mini-batch
        weighted_loss = -weighted_loss.mean()

        return weighted_loss

state_dict = torch.load("/home/arnav/Disk/HistologyNet/bach-contrastive-segmentation/best_metric_model_segmentation2d_array.pth", map_location='cuda')

model_unet.load_state_dict(state_dict)

state_dict_2 = torch.load("/home/arnav/Disk/HistologyNet/TransNuSS/saved_model/best_metric_model_segmentation2d1024.pth", map_location='cuda')

model_unet_2.load_state_dict(state_dict_2)

class Model_(nn.Module):
    def __init__(self, transnuss: nn.Module, color_unet: nn.Module, str_unet: nn.Module):
        super().__init__()
        self.transnuss = transnuss
        self.color_unet = color_unet
        self.str_unet = str_unet
        # self.patch_model.load_state_dict(torch.load(t_checkpoint))
        # self.color_unet.load_state_dict(torch.load(c_checkpoint))
        # self.str_unet.load_state_dict(torch.load(s_checkpoint))
        self.conv_layer = nn.Sequential(nn.Conv2d(4, 8, 3, 1, 1),
                                        nn.ELU(),
                                        nn.BatchNorm2d(8),
                                        nn.Conv2d(8, 4, 3, 1, 1),
                                        nn.ELU(),
                                        nn.BatchNorm2d(4),
                                        nn.Conv2d(4, 1, 3, 1, 1),
                                        nn.Sigmoid())

    def forward(self, x):
        tr_out,_,_,_,_ = self.transnuss(x)
        unet_out = self.color_unet(x)
        st_out = self.str_unet(x)
        return self.conv_layer(torch.cat((tr_out, unet_out, st_out), dim=1))
    
model_ = Model_(transnuss, model_unet, model_unet_2)
model_.to(device)
lr = 1e-4
model_.train()
optimizer = torch.optim.Adam(model_.parameters(), lr=lr)

max_epoch = 30
iterator = tqdm(range(max_epoch))

batch_size = 16
train_loader = DataLoader(db_train_lab, batch_size=batch_size, shuffle=True, num_workers=4)
criterion = nn.BCELoss()
class diceloss(torch.nn.Module):
    def init(self):
        super().init()
    def forward(self,pred, target):
       smooth = 1.
       iflat = pred.contiguous().view(-1)
       tflat = target.contiguous().view(-1)
       intersection = (iflat * tflat).sum()
       A_sum = torch.sum(iflat * iflat)
       B_sum = torch.sum(tflat * tflat)
       return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))
dice = diceloss()
for epoch_num in iterator:
    for iter, sampled_batch in tqdm(enumerate(train_loader)):
        image_batch_normal, label_batch = sampled_batch['image'], sampled_batch['label']
        image_batch_normal, label_batch = image_batch_normal.cuda(), label_batch.cuda()

        output_n = model_(image_batch_normal)
        print(output_n.shape, label_batch.shape)

        loss = criterion(output_n, label_batch) + dice(output_n, label_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Loss: ',loss.item())
        lr = lr * (1.0 - (iter+1) / 200) ** 0.9

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        

    torch.save(model_.state_dict(), 'best_coll.pth')