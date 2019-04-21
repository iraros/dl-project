from pydub.playback import play
from pydub import AudioSegment

import glob
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram

DEFAULT_SR = 22050


def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X, sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds


def plot_waves(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25, 60), dpi=900)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(10, 1, i)
        librosa.display.waveplot(np.array(f), sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 1: Waveplot", x=0.5, y=0.915, fontsize=18)
    plt.show()


def plot_specgram(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25, 60), dpi=900)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(10, 1, i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 2: Spectrogram", x=0.5, y=0.915, fontsize=18)
    plt.show()


def plot_log_power_specgram(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25, 60), dpi=900)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(10, 1, i)
        D = librosa.logamplitude(np.abs(librosa.stft(f)) ** 2, ref_power=np.max)
        librosa.display.specshow(D, x_axis='time', y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 3: Log power spectrogram", x=0.5, y=0.915, fontsize=18)
    plt.show()


def play_array(sound_array):
    temp_path = 'temp wav.wav'
    librosa.output.write_wav(temp_path, sound_array, DEFAULT_SR)
    temp_wav = AudioSegment.from_wav(temp_path)
    play(temp_wav)
    os.remove(temp_path)


# if __name__ == '__main__':
#     sound_file_paths = ["57320-0-0-7.wav", "24074-1-0-3.wav", "15564-2-0-1.wav", "31323-3-0-1.wav",
#                         "46669-4-0-35.wav", "89948-5-0-0.wav", "40722-8-0-4.wav",
#                         "103074-7-3-2.wav", "106905-8-0-0.wav", "108041-9-0-4.wav"]
#
#     sound_names = ["air conditioner", "car horn", "children playing",
#                    "dog bark", "drilling", "engine idling", "gun shot",
#                    "jackhammer", "siren", "street music"]
#
#     raw_sounds = load_sound_files(sound_file_paths)
#
#     plot_waves(sound_names, raw_sounds)
#     plot_specgram(sound_names, raw_sounds)
#     plot_log_power_specgram(sound_names, raw_sounds)

if __name__ == '__main__':
    path = '/home/ira/Desktop/inception-trial/datas/urban sound/fold1 tryouts split by class/dog_bark/7383-3-0-0.wav'
    x, sr = librosa.load(path, mono=False)
    play_array(x)
