import torch
from torch import nn
from torch.nn import Linear
from torchvision import models

# mean and standard deviation for rgb channels of our dataset, computed from 10k samples
RGB_MEAN = [0.6664889994497545, 0.4612734419167714, 0.3871200549055969]
RGB_STD = [0.35835277513458436, 0.4037847429095522, 0.40664457397825116]

DEVICE = (
    "cuda" if torch.cuda.is_available() else "cpu"
)

# NOTE: by default, model expects mini-batches of (3xHxW) with 3RGB channels
model_18 = models.resnet18(weights=None)
model_18.conv1 = nn.Conv2d(3, 64, (7,7), (2,2), (3,3), bias=False)
model_18.fc = Linear(in_features=512, out_features=2)
