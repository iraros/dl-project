# import numpy as np
# import os
# from PIL import Image
# from merger import open_as_band
#
#
# base = '/home/ira/Desktop/dl_project/datas/UrbanSound8K/image_features'
# image_path = 'drilling/161129-4-0-10.wav.png'
# path1 = os.path.join(base, 'recreated_gray_scale_spectrogram_from_channels', image_path)
# path2 = os.path.join(base, 'gray_scale', image_path)
#
#
# channels1 = Image.open(path1).split()
# channels2 = Image.open(path2).split()
#
#
# band1 = open_as_band(path1)
# band2 = open_as_band(path2)
#
# array1 = np.array(band1)
# array2 = np.array(band2)
#
# pass


# coding: utf-8

import csv
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import time


# data_path = '/home/ira/Desktop/inception-trial/datas/UrbanSound8K'
# metadata_path = os.path.join(data_path, 'metadata/UrbanSound8K.csv')

wav_path = '/home/ira/Desktop/dl_project/datas/UrbanSound8K/audio/fold4/344-3-5-0.wav'

y, sr = librosa.load(wav_path)

# Let's make and display a mel-scaled power (energy-squared) spectrogram
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
# S = librosa.feature.spectral_bandwidth(y, sr=sr)

# Convert to log scale (dB). We'll use the peak power as reference.
log_S = librosa.amplitude_to_db(S, ref=np.max)

mfcc_myself = librosa.feature.mfcc(y=y)
mfcc_by_mel_spec = librosa.feature.mfcc(S=S)

# Make a new figure
fig = plt.figure(figsize=(12, 4))
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

# Display the spectrogram on a mel scale
# sample rate and hop length parameters are used to render the time axis
# librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='linear')
# librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='linear')
librosa.display.waveplot(y, cmap='binary', sr=sr, x_axis='time')
# librosa.display.waveplot(y, sr=sr, x_axis='time')

# Make the figure layout compact

plt.show()
# plt.savefig(os.path.join(data_path, thing_name, line[7], line[0] + '.png'))


plt.close()



