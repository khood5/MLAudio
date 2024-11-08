# Recreating the classification setup from Shotspotter

## Paper
- This is based on the following paper:  [Precision and accuracy of acoustic gunshot location in an urban environment](https://arxiv.org/pdf/2108.07377)
    - See section `III. C.` for details


## Notes
- It looks like they have some criteria to determine when a sound is a potential gunshot
- A short clip of audio is taken and used to build the *mosaic* that is fed into the ResNet
- In this recreation, we'll use fields A (waveform), B (discrete wavelength spectrogram) and possibly E and F too
    - Other fields are related to location data, we are focusing on the classifier


## Instructions
- Use `makeBackgroundAudio.py` and `makeGunshotAudio.py` from `MLAudio/data/` to create dataset
    - Make sure to specify sample rate to 12khz (as specified by paper)



## Log
- Tried converting the audio from 2 channels to 1 using mean, completely destroyed the audio
- For now just using setting `-sr` to specify sample rate in make background/gunshot scripts