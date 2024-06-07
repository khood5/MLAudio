from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torchaudio
import os 
import torch

class audioDataloader(Dataset):
    def __init__(
        self,
        index_file: str, 
        header=None,
        transforms=None
    ):
        indexFile = np.array(pd.read_csv(index_file,header=header))
        self.audioFiles = indexFile[:, 0]
        self.audioLabels = indexFile[:, 1]
        self.transforms = transforms
    # frequencies   (Array of sample frequencies):
    #               This represents the array of sample frequencies, i.e., the frequencies at which the spectrogram is calculated.
    #               It corresponds to the y-axis of the spectrogram plot, indicating the different frequency bins.

    # times (Array of segment times):
    #               This represents the array of segment times, i.e., the time points at which each segment of the spectrogram is calculated.
    #               It corresponds to the x-axis of the spectrogram plot, indicating different time points or segments of the signal.

    # spectrogram_data (Spectrogram of x):
    #               This is the actual spectrogram data, representing the magnitude squared of the signal's frequency content at different time segments.
    #               It is a 2D array where the rows correspond to frequency bins (f) and the columns correspond to time segments (t).
    #               The intensity of each element in Sxx represents the magnitude of the frequency component at the corresponding frequency and time.
    def __getitem__(self, index):
        filename = self.audioFiles[index]
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        
        waveform, sample_rate = torchaudio.load(filename, normalize=True)
        mono_waveform = waveform.mean(dim=0)  # Convert stereo to mono
        specgram_transform = torchaudio.transforms.Spectrogram(pad=10)(mono_waveform)
        # Apply additional transforms if provided
        specgram_transform = specgram_transform.unsqueeze(0) # transform from shape (Hight,Width) to (Channel,Hight,Width)
        label_tensor = torch.tensor([self.audioLabels[index]], dtype=torch.float32)
        if self.transforms is not None:
            specgram_transform = self.transforms(specgram_transform)
        return (specgram_transform, label_tensor)

    def getFilePath(self, index):
        return self.audioFiles[index]

    def __len__(self):
        return len(self.audioLabels)