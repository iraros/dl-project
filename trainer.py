import os

import constants
import utils

retrain_command_template = \
    'python retrain.py --image_dir {image_dir} \
    --bottleneck_dir {bottleneck_dir} \
    --output_graph {output_graph} \
    --output_labels {output_labels} \
    --how_many_training_steps={training_steps_num} \
    --testing_percentage 0 \
    --validation_percentage 0'
    # --logging_verbosity


def get_command_string(image_dir, bottleneck_dir, output_graph, output_labels, training_steps_num):
    command_string = retrain_command_template.format(image_dir=image_dir, bottleneck_dir=bottleneck_dir,
                                                     output_graph=output_graph, output_labels=output_labels,
                                                     training_steps_num=training_steps_num)

    return command_string


data_folds_path = '/home/ira/Desktop/dl_project/datas/UrbanSound8K/audio'


# assumes 10 folds of images in data_folds_path
def arrange_by_folds(fold_to_evalute):

    for fold_num in range(1, 10):
        if fold_num == fold_to_evalute:
            continue
        fold_dir = os.path.join(data_folds_path, 'fold' + str(fold_num))
        fold_file_paths = [os.path.join(fold_dir, file_name) for file_name in os.listdir(fold_dir)]


# os.system("python cmd.py 1 21 23  --sum")


# def split_dir_to_class(dir_path):
#     file_names = os.listdir(dir_path)
#     for file_name in file_names:
#         class_num = utils.path2class(file_name)
#         class_name = constants.cls2name[class_num]
#
#     save_dir = os.path.join(dir_path, line[7])
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#

def get_fold_paths():
    base_path = '/home/ira/Desktop/dl_project/datas/UrbanSound8K/audio'
    base_save_path = '/home/ira/Desktop/dl_project/datas/UrbanSound8K/path_mapping'
    for folder_name in os.listdir(base_path):
        if '.' in folder_name:
            continue
        folder = os.path.join(base_path, folder_name)
        paths = [os.path.join(folder, file_name) for file_name in os.listdir(folder)]
        as_string = '\n'.join(paths)
        save_path = os.path.join(base_save_path, folder_name + '.txt')
        open(save_path, 'w+').write(as_string)


get_fold_paths()
