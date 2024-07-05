# TransNuSS

## Usage
Download "R50-ViT-B_16" from https://console.cloud.google.com/storage/vit_models/. Put the downloaded model weights file at "model/vit_checkpoint/imagenet21k/".

### Requirements
- Python 3.6.13
- PyTorch 1.10.2

Install Rest of the requirements using the requirements.txt file provided

#### Fine-tuning dataset
1. Put the dataset at "data/zenodo/"
2. Split the images and masks into "train_images", "train_masks", "validation_images", 
"validation_masks", "test_images", and "test_masks" folder.

Note: You can get the pretrained checkpoints from https://drive.google.com/drive/folders/1AmIL_PaCxUtURSTjvwCpsllTAJHZ6d3F?usp=sharing. Download and place in the /saved_model directory.

### 4. Train
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --root_path data --batch_size 2 --vit_name R50-ViT-B_16
```

### 5. Final Eval
1. Put the two finetuned UNET models in /saved_model/best_unet1.pth and /saved_model/best_unet2.pth
2. Fine Tune the other model
3. Evaluate using:

```bash
python test_ensemble.py --vit_name R50-ViT-B_16
```
