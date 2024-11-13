import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, resample
import numpy as np
import pywt
from PIL import Image
import noisereduce

import argparse
import os

# mosaic resolution constants
OUT_WIDTH = 450
OUT_HEIGHT = 90
DPI_CONST = 200
WIDTH_INCH = OUT_WIDTH / DPI_CONST
HEIGHT_INCH = OUT_HEIGHT / DPI_CONST

# other constants
DWT_LEVELS = 7

parser = argparse.ArgumentParser(description='''generate image mosaics to be fed into resnet.
note that this will join indexes with same name to make it easier to input.''')
parser.add_argument('-s', '--source_dir', required=True, help='source directory with sound files')
parser.add_argument('-o', '--output', required=True, help='directory to output mosaics')
parser.add_argument('-i', '--index_output', required=True, help='where to output index of mosaic paths/labels')
parser.add_argument('-l', '--label', choices=['0', '1'], required=True, help='if sounds in source_dir have gunshots (1) or not (0), used to build index')
args = parser.parse_args()

file_names = os.listdir(args.source_dir)
index_dict = {}
for f in file_names:
    try: 
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
        fig.tight_layout(pad=0.0)

        fig.set_size_inches(WIDTH_INCH, HEIGHT_INCH)

        fig.canvas.draw()
        waveform_graph_pixels = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((OUT_HEIGHT,OUT_WIDTH, 4))
        waveform_graph_pixels = waveform_graph_pixels[:,:,0:3] # exclude alpha channel 

        #plt.savefig(args.output+'/'+raw_f_name+'.png', format='png', dpi=50, bbox_inches='tight', pad_inches=0)
        plt.close()

        # FREQUENCY GRAPH
        data = noisereduce.reduce_noise(y=data, sr=rate)
        # NOTE: we downsample first to reduce time bins to an amount that makes sense
        #data = resample(data, rate)
        # NOTE: we use db1 wavelet family because it perfectly halves at each level and is easier to plot
        coefficients = pywt.wavedec(data, 'db1', level=DWT_LEVELS)
        [print(c.shape) for c in coefficients]
        print()

        fig, ax = plt.subplots(dpi=DPI_CONST)

        # https://stackoverflow.com/questions/64563423/plotting-dwt-scaleogram-in-python
        cc = np.abs(np.array([coefficients[-1]])) # start at coefficient level with most elements
        for i in range(DWT_LEVELS - 1):
            current_coef = coefficients[DWT_LEVELS - 1 - i]
            r = np.abs(np.array([np.repeat(current_coef, pow(2, i + 1))]))
            r = r[:, 0:rate] # tim off little bit at the end, because some of the coefficients don't have exactly previous / 2 elements
            #a = np.abs([cc, r])
            #print(a.shape)
            cc = np.concatenate([cc, r])
            #print(" --- ")

        print(cc.shape)
        
        # X-axis has a linear scale (time)
        x = np.linspace(start=0, stop=1, num=rate*2//2)
        # Y-axis has a logarithmic scale (frequency)
        y = np.linspace(start=DWT_LEVELS-1, stop=0, num=DWT_LEVELS)
        X, Y = np.meshgrid(x, y)
        ax.pcolormesh(X, Y, cc, cmap='copper')

        #ax.pcolormesh(freq_data,  shading='auto', cmap='hot_r')
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
        index_dict[args.output+'/mosaic_'+raw_f_name+'.png'] = args.label

    except Exception as e:
        print('Error: ', e)
        print(f'Skipping {raw_f_name}')
        exit()

# write index
with open(args.index_output, 'a') as f:
    for k, v in index_dict.items():
        f.write(f'{k},{v}\n')