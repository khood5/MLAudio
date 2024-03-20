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
    return resnet18
    
def getBindayClassification():
    resnet18 = models.resnet18()
    resnet18.fc = nn.Sequential(nn.Linear(resnet18.fc.in_features, 1), nn.Sigmoid())# change to binary classification 
    resnet18.conv1 =  nn.Sequential(
                                nn.AvgPool2d((1, 32), stride=(1, 32)), # shrink input 
                                nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # change first conv layer to greyscale (for the spectrogram )
                                ) 
    return resnet18

def main():
    parser = argparse.ArgumentParser(description='Train/Validated ResNet18 on gunshot detection with spectrogram.')
 
    parser.add_argument('valid_dataset', type=str, help='Path to the index file for validation')
    parser.add_argument('model_file', type=str, help='Path to load model params (should be /path/to/FILENAME.pt)')
    parser.add_argument('-d', '--device', default="cuda:0", type=str, help='Device to use, such as cuda:0, cuda:1, cpu, etc. (default: cuda:0 if avalable otherwise cpu)')
    parser.add_argument('-b', '--batch_size', default=10, type=int, help='Batch size for training (default: 10)')
    parser.add_argument('-mt', '--model_type', default='b', type=str, help='Specify the type of resnet18 model to load either Multi-Class (m) or Binday (b) Classification configuration')
    parser.add_argument('-vfn', '--validation_file_name', default='valid.csv', type=str, help='File to output validation loss and accuracies.')
    args = parser.parse_args()

    try:
        device = torch.device(args.device)
        print(f"PyTorch device set to {device}")
    except torch.device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Invalid device. Defaulting to {device}.")
    

    resnet18 = None
    if args.model_type == 'b':
        resnet18 = getBindayClassification()
        print("Model Binday Classification selected and created") 
    else:
        resnet18 = getMulticlassModel()
        print("Model Multi-Class Classification selected and created")
    resnet18.to(device)
    loss_function = nn.CrossEntropyLoss()
    print(f"Model succsefuly loaded to {device}")

    data_transform = transforms.Compose([
        transforms.Normalize(mean=[2.3009], std=[42.1936]) 
    ])

    valid_data = audioDataloader(index_file=args.valid_dataset, transforms=data_transform)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
    print(f"Loaded validation dataset from {args.valid_dataset}")
    
    losses = []
    accuracies = []
    bestLoss = 100 # really big number so first epoch is always smaller 
    print("Starting validation")
    resnet18.load_state_dict(torch.load(args.output_file))
    print(f"model loaded from loss {bestLoss}")
    losses = []
    accuracies = []
    resnet18.eval()
    with tqdm(valid_loader, unit="batch", ncols=96) as tepoch:
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.flatten().type(torch.LongTensor).to(device)
            outputs = resnet18(inputs)
            loss = loss_function(outputs, labels)
            _, predicted = torch.max(torch.round(outputs),1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            losses.append(loss.item())
            accuracy = correct / total
            accuracies.append(100. * accuracy)
            tepoch.set_postfix(loss=f'{loss.item():.3f}', accuracy=f'{(100. * accuracy):.2f}')
            tepoch.update(1)
    with open(args.validation_file_name, 'w', newline ='') as file:    
        write = csv.writer(file)
        write.writerow(["losses","accuracies"])
        write.writerows([list(i) for i in zip(losses,accuracies)])
    print(f"Successfully saved losses and accuracies to {args.valid_file_name}")
    print("Validation done!")

if __name__ == "__main__":
    main()
