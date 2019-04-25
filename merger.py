import os
import re

import numpy as np
from PIL import Image

import utils

sample_wav = '/home/ira/Desktop/dl_project/datas/UrbanSound8K/image_features/wave_forms/dog_bark/344-3-5-0.wav.png'
sample_spc = '/home/ira/Desktop/dl_project/datas/UrbanSound8K/image_features/spectrograms/dog_bark/344-3-5-0.wav.png'


def im_int(image, mult):
    array = np.array(image)
    mult_array = array * mult
    return Image.fromarray(mult_array).convert('L')


def binarize_wave_plot(wave_plot):
    return wave_plot.point(lambda x: 255*int(x < 100))


def open_as_band(path):
    return Image.open(path).convert('L')


empty = Image.new('L', (1200, 400))
full = Image.new('L', (1200, 400), 'white')


def merge_plots(wav_path, spc_path):
    wav = binarize_wave_plot(open_as_band(wav_path))
    spc = open_as_band(spc_path)
    # merged = Image.merge('RGB', [spc, im_int(full, .25), wav])  # base form with slight background
    # merged = Image.merge('RGB', [spc, empty, empty])  # only spectrogram
    merged = Image.merge('RGBA', [spc, spc, wav, full])
    return merged


def stack_plots(wav_path, spc_path):
    wav = Image.open(wav_path)
    spc = Image.open(spc_path)
    w, h = wav.size
    new = Image.new('RGB', (w, h * 2))
    new.paste(spc)
    new.paste(wav, (0, h))
    return new


image_paths_template = '/home/ira/Desktop/dl_project/datas/UrbanSound8K/image_features/{image_type}'


def merge_all_plots(merge_name):
    spectrogram_paths = utils.get_all_paths(image_paths_template.format(image_type='spectrograms'))
    waveplot_paths = utils.get_all_paths(image_paths_template.format(image_type='wave_forms'))
    if len(spectrogram_paths) != len(waveplot_paths):
        raise ValueError
    im_num = len(spectrogram_paths)
    for i in range(im_num):
        waveplot_path = waveplot_paths[i]
        spectrogram_path = spectrogram_paths[i]
        if os.path.basename(waveplot_path) != os.path.basename(spectrogram_path):
            raise ValueError
        # merged_image = merge_plots(waveplot_path, spectrogram_path)
        merged_image = stack_plots(waveplot_path, spectrogram_path)
        save_path = re.sub('wave_forms', merge_name, waveplot_path)
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        merged_image.save(save_path)
        print('finished {} / {}'.format(i, im_num))


if __name__ == '__main__':
    merge_all_plots('spectrogram_and_wav_stacked')
