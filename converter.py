# coding: utf-8
import csv
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import random


data_path = '/home/ira/Desktop/dl_project/datas/UrbanSound8K'
metadata_path = os.path.join(data_path, 'metadata/UrbanSound8K.csv')


def save_all_as_thing(thing_name):
    with open(metadata_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader)  # headers
        for idx, line in enumerate(csv_reader):
            print('arrived: ' + str(idx))
            save_image(line, thing_name)
            print('saved: ' + str(idx))


def save_image(line, thing_name):
    save_dir = os.path.join(data_path, 'image_features', thing_name, line[7])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    y, sr = librosa.load(os.path.join(data_path, 'audio', 'fold' + str(line[5]), str(line[0])))
    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    # Make a new figure
    fig = plt.figure(figsize=(12, 4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    # librosa.display.waveplot(y, cmap='gray_r', sr=sr, x_axis='time')
    # Make the figure layout compact
    # plt.show()
    plt.savefig(os.path.join(save_dir, line[0] + '.png'))
    plt.close()




# save_all_as_thing('small_sample')
