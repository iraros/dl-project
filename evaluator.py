import json
import os
import random

import labeler
import numpy as np

import utils


def evaluate_accuracy(image_paths, true_labels, model_path, labels_path):
    pred_labels = labeler.classify_images(image_paths, model_path, labels_path)
    good_predictions = [true_labels[i] == pred_labels[i] for i in range(len(true_labels))]
    return np.mean(good_predictions), pred_labels


# a dictionary of image_path : class
# roughly equally spread among classes
def create_test_set(base_path, size=.1):
    test_set_dic = {}
    folders = os.listdir(base_path)
    for folder in folders:  # the folder name is the class
        folder_path = os.path.join(base_path, folder)
        files = os.listdir(folder_path)
        chosen_files = random.sample(files, int(len(files) * size))
        chosen_files_paths = {os.path.join(folder_path, chosen_file): folder for chosen_file in chosen_files}
        test_set_dic.update(chosen_files_paths)
    return test_set_dic


def basic_evaluation():
    test_base = '/home/ira/Desktop/dl_project/datas/UrbanSound8K/image_features/gray_scale'
    gray_spectrogram_model_path = '/home/ira/Desktop/dl_project/datas/UrbanSound8K/output_graphs/gray_scale.pb'
    gray_spectrogram_labels_path = '/home/ira/Desktop/dl_project/datas/UrbanSound8K/output_labels/gray_scale.txt'
    test_set = create_test_set(test_base, .01)
    image_paths = list(test_set.keys())
    image_labels = list(test_set.values())
    accuracy = evaluate_accuracy(image_paths, image_labels, gray_spectrogram_model_path, gray_spectrogram_labels_path)
    print(accuracy)


if __name__ == '__main__':
    spectrograms_base_folds = '/home/ira/Desktop/dl_project/datas/UrbanSound8K/image_features/' \
                     'features_for_cross_fold_evaluation/spectrograms'

    base_fold_dir = '/home/ira/Desktop/dl_project/datas/UrbanSound8K/image_features/' \
                    'features_for_cross_fold_evaluation/spectrograms/evaluation_on_fold'
    fold_dirs = [base_fold_dir + str(i) for i in range(1, 11)]

    accuracies_dic = {}
    for evaluation_fold_idx, evaluation_fold_dir in enumerate(fold_dirs):
        if evaluation_fold_idx < 2:
            continue
        evaluation_fold_num = evaluation_fold_idx + 1
        images_dir = os.path.join(evaluation_fold_dir, 'evaluation_fold' + str(evaluation_fold_num))
        model = os.path.join(evaluation_fold_dir, 'model_for_evaluating_fold' + str(evaluation_fold_num) + '.pb')
        labels_file = os.path.join(evaluation_fold_dir, 'model_for_evaluating_fold' + str(evaluation_fold_num) + '.txt')
        performance = os.path.join(evaluation_fold_dir, 'performance_on_fold' + str(evaluation_fold_num) + '.txt')

        images, labels = [], []
        for class_name in os.listdir(images_dir):
            class_dir_path = os.path.join(images_dir, class_name)
            for file in os.listdir(class_dir_path):
                file_path = os.path.join(class_dir_path, file)
                images.append(file_path), labels.append(class_name)
        accuracy, predicted_labels = evaluate_accuracy(images, labels, model, labels_file)

        open(performance, 'w+').write(json.dumps((accuracy, predicted_labels)))
        accuracies_dic[evaluation_fold_dir] = (accuracy, predicted_labels)

        print('FINISHED_FOLD' + str(evaluation_fold_num))

    open(os.path.join(spectrograms_base_folds, 'all_performances.txt'), 'w+').write(json.dumps(accuracies_dic))
