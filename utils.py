import csv
import os
from shutil import copyfile

cls2name = {
    0: 'air_conditioner',
    1: 'car_horn',
    2: 'children_playing',
    3: 'dog_bark',
    4: 'drilling',
    5: 'engine_idling',
    6: 'gun_shot',
    7: 'jackhammer',
    8: 'siren',
    9: 'street_music',
}


def split_dir_by_class(folder_path):
    folder_name = os.path.basename(folder_path)
    parent = os.path.dirname(folder_path)
    split_dir = os.path.join(parent, folder_name + ' split by class')
    class_dirs = [os.path.join(split_dir, class_name) for class_name in cls2name.values()]
    for class_dir in class_dirs:
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

    file_names = os.listdir(folder_path)
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        if '-' not in file_name:
            print(file_name + ' is probably not sound')
            continue
        class_num = int(file_name.split('-')[1])
        class_dir = class_dirs[class_num]
        new_file_path = os.path.join(class_dir, file_name)
        copyfile(file_path, new_file_path)
        print(file_name + ' is a ' + cls2name[class_num])


metadata_path = '/home/ira/Desktop/inception-trial/datas/UrbanSound8K/metadata/UrbanSound8K.csv'
audio_path = '/home/ira/Desktop/inception-trial/datas/UrbanSound8K/audio'
metadata = {}


def name2path(name):
    meta_line = metadata[name]
    fold = meta_line['fold']
    file_path = os.path.join(audio_path, 'fold' + str(fold), name)
    return file_path


def load_meta_data():
    with open(metadata_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        headings = next(csv_reader)
        for idx, line in enumerate(csv_reader):
            metadata[line[0]] = {headings[i]: line[i] for i in range(1, len(line) - 1)}


def path2wav_path(path):
    return name2path(path2wav(path) + '.wav')


def path2wav(path):
    basename = os.path.basename(path)
    return basename.split('.')[0]


def path2class(path):
    wav_name = path2wav(path)
    components = wav_name.split('-')
    class_num = int(components[1])
    class_name = cls2name[class_num]
    return class_name


load_meta_data()
