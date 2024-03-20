import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from audioDataLoader import audioDataloader
from tqdm import tqdm
import argparse
import csv
import numpy as np
from torchvision import transforms

def getMulticlassModel():
    resnet18 = models.resnet18()
    resnet18.fc = nn.Sequential(nn.Linear(resnet18.fc.in_features, 2), nn.Softmax(dim=1))# change to binary classification 
    resnet18.conv1 =  nn.Sequential(
                                nn.AvgPool2d((1, 32), stride=(1, 32)), # shrink input 
                                nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # change first conv layer to greyscale (for the spectrogram )
                                ) 
    return resnet18, nn.CrossEntropyLoss()
    
def getBindayClassification():
    resnet18 = models.resnet18()
    resnet18.fc = nn.Sequential(nn.Linear(resnet18.fc.in_features, 1), nn.Sigmoid())# change to binary classification 
    resnet18.conv1 =  nn.Sequential(
                                nn.AvgPool2d((1, 32), stride=(1, 32)), # shrink input 
                                nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # change first conv layer to greyscale (for the spectrogram )
                                ) 
    return resnet18, nn.BCELoss()