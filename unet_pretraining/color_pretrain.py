import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
from monai.networks.nets import UNet
from monai.data import Dataset

class ColorationDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        colored_image = Image.open(image_path).convert('RGB')  # Load colored image
        grayscale_image = colored_image.convert('L')  # Convert to grayscale
        if self.transform:
            colored_image = self.transform(colored_image)
            grayscale_image = self.transform(grayscale_image)
        return grayscale_image, colored_image
    
if __name__ == '__main__':
    data_dir = 'data/'
    train_data = data_dir + 'train/'
    image_paths = [train_data + 'unlabelled/' + f for f in os.listdir(train_data + 'unlabelled')]
    
    trns = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    train_size = int(0.8 * len(image_paths))
    val_size = len(image_paths) - train_size
    train_dataset, val_dataset = random_split(ColorationDataset(image_paths, transform=trns), [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=2,
        in_channels=1,  # Input channels (grayscale)
        out_channels=3,  # Output channels (RGB)
        channels=(16, 32, 64, 128, 256, 512, 1024),
        strides = (2,2,2,2,2,2)
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    patience = 5 # Number of epochs to wait for improvement
    best_val_loss = float('inf')
    counter = 0

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for i, (grayscale_input, colored_target) in enumerate(train_dataloader, 0):
            grayscale_input, colored_target = grayscale_input.to(device), colored_target.to(device)
            optimizer.zero_grad()

            outputs = model(grayscale_input)
            loss = criterion(outputs, colored_target)  # Compare reconstructed image with original RGB
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (grayscale_input, colored_target) in enumerate(val_dataloader, 0):
                grayscale_input, colored_target = grayscale_input.to(device), colored_target.to(device)

                outputs = model(grayscale_input)
                loss = criterion(outputs, colored_target)  # Compare reconstructed image with original RGB
                val_loss += loss.item()

        print('[%d] Training Loss: %.6f, Validation Loss: %.6f' % (epoch + 1, running_loss / len(train_dataloader), val_loss / len(val_dataloader)))

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping after epoch {epoch + 1}')
                break

    print('Finished Pre Training')
    print("Saving model...")
    torch.save(model.state_dict(), '/model/color_pretrain.pth')
    