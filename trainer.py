import os
import shutil

import utils

retrain_command_template = \
    'python retrain.py --image_dir {image_dir} \ \n \
    --bottleneck_dir {bottleneck_dir} \ \n \
    --output_graph {output_graph} \ \n \
    --output_labels {output_labels} \ \n \
    --how_many_training_steps={training_steps_num} \ \n \
    --testing_percentage 1 \ \n \
    --test_batch_size 1 \ \n \
    --validation_percentage 1 \ \n \
    --validation_batch_size 1'

# os.system("python cmd.py 1 21 23  --sum")


SPECTROGRAM_BOTTLENECK_DIR = '/home/ira/Desktop/dl_project/datas/UrbanSound8K/bottlenecks/spectrograms'


def get_command_string(image_dir, bottleneck_dir, output_graph, output_labels, training_steps_num):
    command_string = retrain_command_template.format(image_dir=image_dir, bottleneck_dir=bottleneck_dir,
                                                     output_graph=output_graph, output_labels=output_labels,
                                                     training_steps_num=training_steps_num)
    return command_string


def train_model(image_dir, bottleneck_dir, output_graph, output_labels, training_steps_num):
    command_string = get_command_string(image_dir, bottleneck_dir, output_graph, output_labels, training_steps_num)
    open('/home/ira/Desktop/dl_project/datas/UrbanSound8K/image_features/features_for_cross_fold_evaluation/'
         'spectrograms/training_scripts.txt', mode='a+').write(command_string + '\n\n')


def split_to_evaluation_dirs_by_fold():
    spectrogram_dir = '/home/ira/Desktop/dl_project/datas/UrbanSound8K/image_features/spectrograms'
    evaluation_features_base_path = '/home/ira/Desktop/dl_project/datas/UrbanSound8K/image_features/' \
                                    'features_for_cross_fold_evaluation/spectrograms'
    spc_fold2path = utils.get_paths_in_dir_by_fold(spectrogram_dir)
    fold_dirs = [os.path.join(evaluation_features_base_path, 'evaluation_on_fold' + str(i)) for i in range(1, 11)]
    for evaluation_fold_idx, evaluation_fold_dir in enumerate(fold_dirs):
        evaluation_fold_num = evaluation_fold_idx + 1
        os.makedirs(evaluation_fold_dir)

        evaluation_files = spc_fold2path[evaluation_fold_num]
        training_files_lists = [spc_fold2path[fold] for fold in spc_fold2path.keys() if fold != evaluation_fold_num]
        training_files = [file for sublist in training_files_lists for file in sublist]

        training_dir = os.path.join(evaluation_fold_dir, 'training_all_folds_but_fold' + str(evaluation_fold_num))
        evaluation_dir = os.path.join(evaluation_fold_dir, 'evaluation_fold' + str(evaluation_fold_num))
        os.makedirs(training_dir)
        os.makedirs(evaluation_dir)
        for evaluation_file in evaluation_files:
            base_name = os.path.basename(evaluation_file)
            class_name = os.path.basename(os.path.dirname(evaluation_file))
            class_dir = os.path.join(evaluation_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            destination_file = os.path.join(class_dir, base_name)
            shutil.copyfile(evaluation_file, destination_file)
        for training_file in training_files:
            base_name = os.path.basename(training_file)
            class_name = os.path.basename(os.path.dirname(training_file))
            class_dir = os.path.join(training_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            destination_file = os.path.join(class_dir, base_name)
            shutil.copyfile(training_file, destination_file)


TRAINING_STEPS_DEFAULT_NUM = 10000


def print_retraining_commands():
    evaluation_features_base_path = '/home/ira/Desktop/dl_project/datas/UrbanSound8K/image_features/' \
                                    'features_for_cross_fold_evaluation/spectrograms'
    fold_dirs = [os.path.join(evaluation_features_base_path, evaluation_fold_dir) for evaluation_fold_dir in
                 os.listdir(evaluation_features_base_path)]

    for evaluation_fold_idx, evaluation_fold_dir in enumerate(fold_dirs):
        evaluation_fold_num = evaluation_fold_idx + 1

        training_dir = os.path.join(evaluation_fold_dir, 'training_all_folds_but_fold' + str(evaluation_fold_num))
        # evaluation_dir = os.path.join(evaluation_fold_dir, 'evaluation_fold' + str(evaluation_fold_num))

        # bottle_neck_path = os.path.join(evaluation_fold_dir, 'bottleneck_for_evaluating_fold' + str(evaluation_fold_num))
        model_save_path = os.path.join(evaluation_fold_dir,
                                       'model_for_evaluating_fold' + str(evaluation_fold_num) + '.pb')
        labels_save_path = os.path.join(evaluation_fold_dir,
                                        'model_for_evaluating_fold' + str(evaluation_fold_num) + '.txt')

        train_model(training_dir, SPECTROGRAM_BOTTLENECK_DIR, model_save_path, labels_save_path,
                    TRAINING_STEPS_DEFAULT_NUM)


if __name__ == '__main__':
    print_retraining_commands()
    # split_to_evaluation_dirs_by_fold()
