import matplotlib.pyplot as plt
from scipy.io import wavfile
#from scipy.signal import resample
import numpy as np

import argparse
import os

parser = argparse.ArgumentParser(description='generate image mosaics to be fed into resnet')
parser.add_argument('-s', '--source_dir', required=True, help='source directory with sound files')
parser.add_argument('-o', '--output', required=True, help='directory to output mosaics')
args = parser.parse_args()

file_names = os.listdir(args.source_dir)
for f in file_names:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html
    print(args.source_dir+'/'+f)
    rate, data = wavfile.read(args.source_dir+'/'+f)
    if len(data.shape) == 2: # if 2 channels 
        data = data[:, 0] # take only left channel

    # downsampling (maybe for later)
    # we want to downsample to 12khz like the paper
    #target_rate = 12000
    #data = resample(data, num=int(len(data) * target_rate / rate))
    #wavfile.write('downsampled.wav', target_rate, data)

    time = np.arange(data.shape[0])/rate

    fig, ax = plt.subplots()
    ax.plot(time, data, color='g')
    ax.axis('off')

    width_inch = 450 / 50
    height_inch = 90 / 50
    fig.set_size_inches(width_inch, height_inch)

    plt.tight_layout(pad=0.0)
    plt.savefig(args.output+'/'+f+'.png', format='png', dpi=50, bbox_inches='tight', pad_inches=0)
    plt.close()