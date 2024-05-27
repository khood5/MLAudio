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
from models import getBinaryResNetModel, getMulticlassResNetModel

def main():
    parser = argparse.ArgumentParser(description='Validated ResNet18 on gunshot detection with spectrogram.')
 
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
    loss_function = None
    if args.model_type == 'b':
        resnet18, loss_function = getBinaryResNetModel()
        print("Model Binday Classification selected and created") 
    else:
        resnet18, loss_function = getMulticlassResNetModel()
        print("Model Multi-Class Classification selected and created")
    resnet18.to(device)
    print(f"Model succsefuly loaded to {device}")

    data_transform = transforms.Compose([
        transforms.Normalize(mean=[2.3009], std=[42.1936]),
        TransposeTensor(),
    ])

    valid_data = audioDataloader(index_file=args.valid_dataset, transforms=data_transform)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
    print(f"Loaded validation dataset from {args.valid_dataset}")
    
    losses = []
    accuracies = []
    print("Starting validation")
    resnet18.load_state_dict(torch.load(args.model_file))
    print(f"Model succsefuly loaded from file {args.model_file}")
    losses = []
    accuracies = []
    resnet18.eval()
    with tqdm(valid_loader, unit="batch", ncols=96) as tepoch:
        for inputs, labels in valid_loader:
            if args.model_type == 'b':
                inputs, labels = inputs.to(device), labels.to(device)
            else:
                inputs, labels = inputs.to(device), labels.flatten().type(torch.LongTensor).to(device)
            outputs = resnet18(inputs)
            loss = loss_function(outputs, labels)
            losses.append(loss.item())

            total = labels.size(0)
            if args.model_type == 'b':
                predicted = torch.round(outputs)
            else:
                _, predicted = torch.max(torch.round(outputs),1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / total
            accuracies.append(100. * accuracy)
            
            tepoch.set_postfix(loss=f'{loss.item():.3f}', accuracy=f'{(100. * accuracy):.2f}')
            tepoch.update(1)
    with open(args.validation_file_name, 'w', newline ='') as file:    
        write = csv.writer(file)
        write.writerow(["losses","accuracies"])
        write.writerows([list(i) for i in zip(losses,accuracies)])
    print(f"Successfully saved losses and accuracies to {args.validation_file_name}")
    print("Validation done!")

# swap freq and time dim to match SNN
class TransposeTensor(object):
    def __call__(self, tensor):
        return torch.transpose(tensor, 1, 2)

if __name__ == "__main__":
    main()
