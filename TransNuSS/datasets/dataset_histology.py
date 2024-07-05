import glob
import os
import numpy as np
import scipy.io
from pathlib import Path
import torch
import pickle
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms.transforms import Grayscale, RandomCrop

# class Histology(Dataset):
#     def __init__(self, image_directory: str, epsilon: float = 0.05,crop_size: int = 256):
#         image_files = glob.glob(os.path.join(image_directory, 'images', '*.png'))
#         images = map(Image.open, image_files)
#         images = map(ToTensor(), images)
#         self.images = list(images)

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         image = self.images[idx]
#         return image
mean_std_file = './data/zenodo/zenodo_mean_std.txt'

with open(mean_std_file, 'rb') as handle:
    data_mean_std = pickle.loads(handle.read())

class Histology(Dataset):
    def __init__(self, image_directory: str, crop_size: int = 256):
        self.image_dir = Path(image_directory)
        self.image_paths = [str(path) for path in self.image_dir.glob("*.png")]  # Get image paths
        self.cropper = RandomCrop(crop_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = np.array(image).astype('float32')
        image = ((image - data_mean_std['mean_train_images']) / data_mean_std['std_train_images'])
        image = ToTensor()(image)
        return image, image_path


class HistologyValidation(Dataset):
    def __init__(self, directory: str, epsilon: float = 0.05):
        self.image_dir = Path(os.path.join(directory, 'images'))
        self.masks_dir = Path(os.path.join(directory, 'masks'))
        image_files = glob.glob(os.path.join(directory, 'images', '*.png'))
        mask_files = [
            os.path.join(directory, "masks", os.path.basename(image_file))
            for image_file in image_files
        ]

        self.image_paths = [str(path) for path in self.image_dir.glob("*.png")]
        self.mask_paths = [str(path) for path in self.masks_dir.glob("*.png")]

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = np.array(image).astype('float32')
        image = ((image - data_mean_std['mean_train_images']) / data_mean_std['std_train_images'])
        image = ToTensor()(image)

        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path)
        mask = np.array(mask.convert('1')).astype('uint8')
        mask = ToTensor()(mask)
        return image, mask, mask_path


