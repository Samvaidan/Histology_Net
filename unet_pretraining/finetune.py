import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.transforms import RandomVerticalFlip, ColorJitter, RandomHorizontalFlip, Compose, Resize, ToTensor
from monai.losses import DiceLoss
from PIL import Image
from monai.networks.nets import UNet
from monai.data import Dataset
import logging
import os
import sys
import tempfile
from glob import glob
import torch
from PIL import Image
import monai
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch, DataLoader, list_data_collate
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    ScaleIntensityd,
    LoadImaged,
    EnsureChannelFirstd,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
)
from monai.visualize import plot_2d_or_3d_image



class SegmentationDataset(Dataset):
    def __init__(self, image_files, mask_files, transform=None):
        self.image_files = image_files
        self.mask_files = mask_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return {'image': image, 'mask': mask}



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrained_state = torch.load('/model/color_pretrain2.pth')
    
    model3 = UNet(
    spatial_dims=2,
    in_channels=3, 
    out_channels=1, 
    channels=(16, 32, 64, 128, 256, 512, 1024),
    strides=(2, 2, 2, 2, 2, 2)
    ).to(device)
    
    for name, param in model3.named_parameters():
        if name in pretrained_state and param.shape == pretrained_state[name].shape:
            param.data.copy_(pretrained_state[name]) 
            
    #Model 3 created
    #Now finetuning
    train_transforms = Compose([
        Resize((256, 256)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ToTensor()
    ])

    val_transforms = Compose([
        Resize((256, 256)),
        ToTensor()
    ])
    
    data_dir = 'data'
    train_images = sorted(glob(os.path.join(data_dir, 'train', 'labelled','images',"*.png")))
    train_segs = sorted(glob(os.path.join(data_dir,'train','labelled','masks', "*.png")))
    validation_images = sorted(glob(os.path.join(data_dir, 'validation', 'labelled','images',"*.png")))
    validation_segs = sorted(glob(os.path.join(data_dir,'validation','labelled','masks', "*.png")))
    
    
    def collate_fn(batch):
      return {
          'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
          'labels': torch.tensor([x['labels'] for x in batch])
    }
      
        
    train_files = [{"img": img, "seg": seg} for img, seg in zip(train_images, train_segs)]
    val_files = [{"img": img, "seg": seg} for img, seg in zip(validation_images, validation_segs)]

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys=["img", "seg"]),
    #         RandCropByPosNegLabeld(
    #             keys=["img", "seg"], label_key="seg", spatial_size=[256, 256], pos=1, neg=1, num_samples=4
    #         ),
    #         RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys=["img", "seg"]),
        ]
    )

  
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
  
    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
        # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    # create UNet, DiceLoss and Adam optimizer
    # model.load_state_dict(torch.load('best_metric_model_segmentation2d_dict_new.pth'))
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model3.parameters(), 1e-4, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for binary segmentation

        # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    for epoch in range(50):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{10}")
        model3.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            outputs = model3(inputs)
            bce = criterion(outputs, labels)
            loss = loss_function(outputs, labels)
            w_bce = 0.3
            loss2 = w_bce * bce + (1 - w_bce) * (loss)
            
            loss2.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
    #         print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model3.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                    roi_size = (256, 256)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model3)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model3.state_dict(), f"best_metric_model_segmentation2d1024_{epoch}.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    
    #Saving model
    torch.save(model3.state_dict(), '/model/fine_tuned_model.pth')



      
    
    


            
            