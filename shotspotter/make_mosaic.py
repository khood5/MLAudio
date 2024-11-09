import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram #resample
import numpy as np
import pywt
from PIL import Image

import argparse
import os

OUT_WIDTH = 300
OUT_HEIGHT = 60
DPI_CONST = 100
WIDTH_INCH = OUT_WIDTH / DPI_CONST
HEIGHT_INCH = OUT_HEIGHT / DPI_CONST

parser = argparse.ArgumentParser(description='generate image mosaics to be fed into resnet')
parser.add_argument('-s', '--source_dir', required=True, help='source directory with sound files')
parser.add_argument('-o', '--output', required=True, help='directory to output mosaics')
args = parser.parse_args()

file_names = os.listdir(args.source_dir)
for f in file_names:
    raw_f_name = f.replace('.wav', '')

    print('reading ' + args.source_dir+'/'+f)
    rate, data = wavfile.read(args.source_dir+'/'+f)
    if len(data.shape) == 2: # if 2 channels 
        data = data[:, 0] # take only left channel

    # WAVEFORM GRAPH
    time = np.arange(data.shape[0])/rate

    fig, ax = plt.subplots(dpi=DPI_CONST)
    ax.plot(time, data, color='g')
    ax.axis('off')

    fig.set_size_inches(WIDTH_INCH, HEIGHT_INCH)
    plt.tight_layout(pad=0.0)

    fig.canvas.draw()
    waveform_graph_pixels = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((OUT_HEIGHT,OUT_WIDTH, 4))
    waveform_graph_pixels = waveform_graph_pixels[:,:,0:3] # exclude alpha channel 

    #plt.savefig(args.output+'/'+raw_f_name+'.png', format='png', dpi=50, bbox_inches='tight', pad_inches=0)
    plt.close()

    # FREQUENCY GRAPH
    freq_data = pywt.wavedec(data, 'haar', level=5)

    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
    # note this function might be removed
    f, t, s = spectrogram(freq_data[1], fs=rate, nperseg=32)

    fig, ax = plt.subplots(dpi=DPI_CONST)
    ax.pcolormesh(t, f, 10*np.log10(s), shading='auto', cmap='OrRd')
    ax.axis('off')
    fig.tight_layout(pad=0.0)

    fig.set_size_inches(WIDTH_INCH, HEIGHT_INCH)

    fig.canvas.draw()
    spectro_graph_pixes = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((OUT_HEIGHT, OUT_WIDTH, 4))
    spectro_graph_pixes = spectro_graph_pixes[:,:,0:3]

    #plt.savefig(args.output+'/freq_'+raw_f_name+'.png', format='png', dpi=50)
    plt.close()

    # MAKE MOSAIC
    output = Image.new("RGB", (OUT_WIDTH, OUT_HEIGHT*2))
    output.paste(Image.fromarray(waveform_graph_pixels), (0,0))
    output.paste(Image.fromarray(spectro_graph_pixes), (0, OUT_HEIGHT))

    output.save(args.output+'/mosaic_'+raw_f_name+'.png')