import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from audioDataLoader import audioDataloader
from tqdm import tqdm
import argparse
import csv


def main():
    parser = argparse.ArgumentParser(description='Train/Validated ResNet18 on gunshot detection with spectrogram.')
 
    parser.add_argument('train_dataset', type=str, help='Path to the index file for training data')
    parser.add_argument('valid_dataset', type=str, help='Path to the index file for validation')
    parser.add_argument('output_file', type=str, help='Path to save model params (should be /path/to/FILENAME.pt)')
    parser.add_argument('-d', '--device', default="cuda:0", type=str, help='Device to use, such as cuda:0, cuda:1, cpu, etc. (default: cuda:0 if avalable otherwise cpu)')
    parser.add_argument('-b', '--batch_size', default=10, type=int, help='Batch size for training (default: 10)')
    parser.add_argument('-e', '--epoch', default=25, type=int, help='Number of epochs for training (default: 25)')
    args = parser.parse_args()

    try:
        device = torch.device(args.device)
        print(f"PyTorch device set to {device}")
    except torch.device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Invalid device. Defaulting to {device}.")
    

    resnet18 = models.resnet18()
    resnet18.fc = nn.Sequential(nn.Linear(resnet18.fc.in_features, 1), nn.Sigmoid())# change to binary classification 
    resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # change input layer to greyscale (for the spectrogram )
    loss_function = nn.BCELoss()
    optimizer = optim.SGD(resnet18.parameters(), lr=0.1, weight_decay=0.0001, momentum=0.9)
    resnet18.to(device)
    print(f"Model succsefuly made and loaded to {device}")

    train_data = audioDataloader(index_file=args.train_dataset)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    print(f"Loaded training dataset from {args.train_dataset}")
    valid_data = audioDataloader(index_file=args.valid_dataset)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
    print(f"Loaded validation dataset from {args.valid_dataset}")
    
    losses = []
    accuracies = []
    for epoch in range(args.epoch):
        print(f"Epoch {epoch}")
        resnet18.train()
        with tqdm(train_loader, unit="batch", ncols=128) as tepoch:
            for inputs, labels in train_loader:
                inputs, labels = torch.unsqueeze(inputs, 1).to(device), torch.unsqueeze(labels, 1).type(torch.float32).to(device)
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
                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
                if len(losses) > 1 and losses[-1] < losses[-2]:
                        torch.save(resnet18.state_dict(), args.output_file)
    file = open('train.csv', 'w', newline ='')
    with file:    
        write = csv.writer(file)
        write.writerow(["losses","accuracies"])
        write.writerows([list(i) for i in zip(losses,accuracies)])
    print(f"Successfully saved losses and accuracies to train.csv")
    print("Training done!")

    print("Starting validation")
    losses = []
    accuracies = []
    for epoch in range(args.epoch):
        print(f"Epoch {epoch}")
        resnet18.eval()
        with tqdm(valid_loader, unit="batch") as tepoch:
            for inputs, labels in valid_loader:
                inputs, labels = torch.unsqueeze(inputs, 1).to(device), torch.unsqueeze(labels, 1).type(torch.float32).to(device)
                outputs = resnet18(inputs)
                loss = loss_function(outputs, labels)
                predicted = torch.round(outputs)
                total = labels.size(0)
                correct = (predicted == labels).sum().item()
                losses.append(loss.item())
                accuracy = correct / total
                accuracies.append(100. * accuracy)
                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
    file = open('validation.csv', 'w', newline ='')
    with file:    
        write = csv.writer(file)
        write.writerow(["losses","accuracies"])
        write.writerows([list(i) for i in zip(losses,accuracies)])
    print(f"Successfully saved losses and accuracies to validation.csv")
    print("Validation done!")

if __name__ == "__main__":
    main()
