# author: Louca Malerba
# date: 16/08/2025

# Public libraries importations
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
import sys

import kcaptcha.datasets.dataset as dataset

# 1. Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Setup seed
torch.manual_seed(42) 
torch.cuda.manual_seed(42)
random.seed(42)

# 3. Setting up hyperparameters
BATCH_SIZE = 64
EPOCHS = 75
LEARNING_RATE = 1e-4
NUM_CLASSES = 10
IMAGE_WIDTH = 40
IMAGE_LENGTH = 40
PATCH_WIDTH, PATCH_LENGTH = 8,8 
CHANNELS = 3
EMBED_DIM = 512
NUM_HEADS = 8
DEPTH = 8
MLP_DIM = 512
DROP_RATE = 0.1

# 5. Import dataset
path_to_dataset = '/Users/loucamalerba/Desktop/captcha_dataset_detection/KCaptcha_raw/2digits_nonoise/images'

transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_LENGTH, IMAGE_WIDTH)),
    # transforms.RandomHorizontalFlip(0.3),
    # transforms.RandomVerticalFlip(0.3),
    # transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(((0.5), (0.5), (0.5)), ((0.5), (0.5), (0.5))),
    
])

test_transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),
    # transforms.RandomHorizontalFlip(0.3),
    # transforms.RandomVerticalFlip(0.3),
    # transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(((0.5), (0.5), (0.5)), ((0.5), (0.5), (0.5))),
    
])


train_dataset = dataset.ImageCaptchaDataset(root=path_to_dataset,
                                 train=True,
                                 img_h=IMAGE_LENGTH, img_w=IMAGE_WIDTH,
                                 transform=transform)


test_dataset = dataset.ImageCaptchaDataset(root=path_to_dataset,
                                 train=False,
                                 img_h=IMAGE_LENGTH, img_w=IMAGE_WIDTH,
                                 transform=test_transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False)

criterion = nn.CTCLoss(blank=0, zero_infinity=True)


