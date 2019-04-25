import csv
import os
import re
import numpy as np
from shutil import copyfile

import constants


def split_dir_by_class(folder_path):
    folder_name = os.path.basename(folder_path)
    parent = os.path.dirname(folder_path)
    split_dir = os.path.join(parent, folder_name + ' split by class')
    class_dirs = [os.path.join(split_dir, class_name) for class_name in constants.cls2name.values()]
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
        print(file_name + ' is a ' + constants.cls2name[class_num])


metadata_path = '/home/ira/Desktop/dl_project/datas/UrbanSound8K/metadata/UrbanSound8K.csv'
audio_path = '/home/ira/Desktop/dl_project/datas/UrbanSound8K/audio'
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
    return name2path(path2name(path) + '.wav')


def path2name(path):
    basename = os.path.basename(path)
    return basename.split('.')[0]


def path2class(path):
    name = path2name(path)
    components = name.split('-')
    class_num = int(components[1])
    class_name = constants.cls2name[class_num]
    return class_name


def clean_misclassifications_file(file_path):
    text = open(file_path).read()
    text = re.sub('INFO:.+', '', text)
    text = re.sub('.+(?=datas)', '', text)
    text = re.sub('\n\n', '\n', text)
    lines = text.split('\n')
    not_empty_lines = [line for line in lines if line.strip()]
    open(file_path, 'w').write('\n'.join(not_empty_lines))


def get_mean_length():
    load_meta_data()
    lengths = []
    for name, data in metadata.items():
        length = float(data['end']) - float(data['start'])
        lengths.append(length)
    len_arr = np.array(lengths)
    mean_length = np.mean(len_arr)
    print(mean_length)  # 3.61


def get_all_paths(root):
    paths = []
    for path, sub_dirs, files in os.walk(root):
        for name in files:
            paths.append(os.path.join(path, name))
    return paths


def path2fold(path):
    name = path2name(path)
    file_metadata = metadata[name + '.wav']
    return int(file_metadata['fold'])


def get_paths_in_dir_by_fold(dir_path):
    paths_in_dir = get_all_paths(dir_path)
    path2fold_dic = {path: path2fold(path) for path in paths_in_dir}
    fold2path_dic = {}
    for path, fold in path2fold_dic.items():
        fold2path_dic[fold] = fold2path_dic.get(fold, [])
        fold2path_dic[fold].append(path)
    return fold2path_dic


load_meta_data()


# if __name__ == '__main__':
#     result = get_paths_in_dir_by_fold('/home/ira/Desktop/dl_project/datas/UrbanSound8K/image_features/spectrograms')
#     pass
