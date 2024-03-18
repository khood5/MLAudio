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

def main():
    parser = argparse.ArgumentParser(description='Train/Validated ResNet18 on gunshot detection with spectrogram.')
 
    parser.add_argument('train_dataset', type=str, help='Path to the index file for training data')
    parser.add_argument('valid_dataset', type=str, help='Path to the index file for validation')
    parser.add_argument('output_file', type=str, help='Path to save model params (should be /path/to/FILENAME.pt)')
    parser.add_argument('-d', '--device', default="cuda:0", type=str, help='Device to use, such as cuda:0, cuda:1, cpu, etc. (default: cuda:0 if avalable otherwise cpu)')
    parser.add_argument('-b', '--batch_size', default=10, type=int, help='Batch size for training (default: 10)')
    parser.add_argument('-e', '--epoch', default=25, type=int, help='Number of epochs for training (default: 25)')
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float, help='Learning rate (default: 0.1')
    parser.add_argument('-wd', '--weight_decay', default=0.0001, type=float, help='Weight decay (default:  0.0001')
    parser.add_argument('-m', '--momentum', default=0.9, type=float, help='Momentum (default:  0.9')
    parser.add_argument('-tfn', '--train_file_name', default='train.csv', type=str, help='file to output training loss and accuracies')
    parser.add_argument('-vfn', '--valid_file_name', default='validation.csv', type=str, help='file to output validation loss and accuracies')
    args = parser.parse_args()

    try:
        device = torch.device(args.device)
        print(f"PyTorch device set to {device}")
    except torch.device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Invalid device. Defaulting to {device}.")
    

    resnet18 = models.resnet18()
    resnet18.fc = nn.Sequential(nn.Linear(resnet18.fc.in_features, 1), nn.Sigmoid())# change to binary classification 
    resnet18.conv1 =  nn.Sequential(
                                nn.AvgPool2d((1, 32), stride=(1, 32)), # shrink input 
                                nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # change first conv layer to greyscale (for the spectrogram )
                                ) 
    loss_function = nn.BCELoss()
    optimizer = optim.SGD(resnet18.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
    resnet18.to(device)
    print(f"Model succsefuly made and loaded to {device}")

    data_transform = transforms.Compose([
        transforms.Normalize(mean=[2.3009], std=[42.1936]) 
    ])

    train_data = audioDataloader(index_file=args.train_dataset, transforms=data_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    print(f"Loaded training dataset from {args.train_dataset}")
    valid_data = audioDataloader(index_file=args.valid_dataset, transforms=data_transform)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
    print(f"Loaded validation dataset from {args.valid_dataset}")
    
    losses = []
    accuracies = []
    for epoch in range(args.epoch):
        print(f"Epoch {epoch}")
        resnet18.train()
        with tqdm(train_loader, unit="batch", ncols=96) as tepoch:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = resnet18(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                predicted = torch.round(outputs)
                total = labels.size(0)
                correct = (predicted == labels).sum().item()
                losses.append(loss.item())
                accuracy = correct / total
                accuracies.append(100. * accuracy)
                tepoch.set_postfix(loss=f'{loss.item():.3f}', accuracy=f'{(100. * accuracy):.2f}')
                tepoch.update(1)
        torch.save(resnet18.state_dict(), args.output_file)
        print("Saved model")
        print(f"Epoch {epoch} summary:")
        print(f"mean accuracy {np.mean(accuracies):.2f}")
        print(f"mean loss of  {np.mean(losses):.3f}")
        print()
                
    file = open(args.train_file_name, 'w', newline ='')
    with file:    
        write = csv.writer(file)
        write.writerow(["losses","accuracies"])
        write.writerows([list(i) for i in zip(losses,accuracies)])
    print(f"Successfully saved losses and accuracies to {args.train_file_name}")
    print("Training done!")

    print("Starting validation")
    losses = []
    accuracies = []
    resnet18.eval()
    with tqdm(valid_loader, unit="batch", ncols=96) as tepoch:
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = resnet18(inputs)
            loss = loss_function(outputs, labels)
            predicted = torch.round(outputs)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            losses.append(loss.item())
            accuracy = correct / total
            accuracies.append(100. * accuracy)
            tepoch.set_postfix(loss=f'{loss.item():.3f}', accuracy=f'{(100. * accuracy):.2f}')
            tepoch.update(1)
    file = open(args.valid_file_name, 'w', newline ='')
    with file:    
        write = csv.writer(file)
        write.writerow(["losses","accuracies"])
        write.writerows([list(i) for i in zip(losses,accuracies)])
    print(f"Successfully saved losses and accuracies to {args.valid_file_name}")
    print("Validation done!")

if __name__ == "__main__":
    main()
