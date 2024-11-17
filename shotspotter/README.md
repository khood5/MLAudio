# Recreating the classification setup from Shotspotter

## Paper
- This is based on the following paper:  [Precision and accuracy of acoustic gunshot location in an urban environment](https://arxiv.org/pdf/2108.07377)
    - See section `III. C.` for details


## Notes
- It looks like they have some criteria to determine when a sound is a potential gunshot
- A short clip of audio is taken and used to build the *mosaic* that is fed into the ResNet
- In this recreation, we'll use fields A (waveform), B (discrete wavelength spectrogram) and possibly E and F too
    - Other fields are related to location data, we are focusing on the classifier
- Paper classifies on 2-second 12khz clips of audio, we can create these using the scripts
- Try to match resolution or level of detail on paper's mosaic
    - See Instructions / Configuration below


## Instructions / Configuration
- Use `makeBackgroundAudio.py` and `makeGunshotAudio.py` from `MLAudio/data/` to create dataset
    - Make sure to specify sample rate to 12khz (as specified by paper)
- make_mosaic has some constants that can be changed to change output mosaic size

## Log
- Tried converting the audio from 2 channels to 1 using mean, completely destroyed the audio
- For now just using setting `-sr` to specify sample rate in make background/gunshot scripts
- Spectrogram and waveform look reasonably close to paper
- For now denoising is done using `noisereduce` library, work on this more