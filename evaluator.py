import json
import os

import constants
import labeler
import numpy as np

spectrograms_base_folds = '/home/ira/Desktop/dl_project/datas/UrbanSound8K/image_features/' \
                          'features_for_cross_fold_evaluation/spectrograms'
base_fold_dir = '/home/ira/Desktop/dl_project/datas/UrbanSound8K/image_features/' \
                'features_for_cross_fold_evaluation/spectrograms/evaluation_on_fold'
fold_dirs = [base_fold_dir + str(i) for i in range(1, 11)]


def evaluate_accuracy(image_paths, true_labels, model_path, labels_path):
    pred_labels = labeler.classify_images(image_paths, model_path, labels_path)
    good_predictions = [true_labels[i] == pred_labels[i] for i in range(len(true_labels))]
    return np.mean(good_predictions), pred_labels


def evaluate_cross_folds():
    accuracies_dic = {}
    for evaluation_fold_idx, evaluation_fold_dir in enumerate(fold_dirs):
        if evaluation_fold_idx < 5:
            continue
        evaluation_fold_num = evaluation_fold_idx + 1
        images_dir = os.path.join(evaluation_fold_dir, 'evaluation_fold' + str(evaluation_fold_num))
        model = os.path.join(evaluation_fold_dir, 'model_for_evaluating_fold' + str(evaluation_fold_num) + '.pb')
        labels_file = os.path.join(evaluation_fold_dir, 'model_for_evaluating_fold' + str(evaluation_fold_num) + '.txt')
        performance = os.path.join(evaluation_fold_dir, 'performance_on_fold' + str(evaluation_fold_num) + '.txt')

        images, labels = get_images_and_labels(images_dir)
        accuracy, predicted_labels = evaluate_accuracy(images, labels, model, labels_file)

        open(performance, 'w+').write(json.dumps((accuracy, predicted_labels)))
        accuracies_dic[evaluation_fold_dir] = (accuracy, predicted_labels)

        print('FINISHED_FOLD' + str(evaluation_fold_num))
    open(os.path.join(spectrograms_base_folds, 'all_performances.txt'), 'w+').write(json.dumps(accuracies_dic))


def get_images_and_labels(images_dir):
    images, labels = [], []
    for class_name in os.listdir(images_dir):
        class_dir_path = os.path.join(images_dir, class_name)
        for file in os.listdir(class_dir_path):
            file_path = os.path.join(class_dir_path, file)
            images.append(file_path), labels.append(class_name)
    return images, labels


def read_performance_txts():
    accuracies, true_labels, predicted_labels = [], [], []
    for evaluation_fold_idx, evaluation_fold_dir in enumerate(fold_dirs):
        evaluation_fold_num = evaluation_fold_idx + 1
        images_dir = os.path.join(evaluation_fold_dir, 'evaluation_fold' + str(evaluation_fold_num))
        performance = os.path.join(evaluation_fold_dir, 'performance_on_fold' + str(evaluation_fold_num) + '.txt')

        _, dir_true_labels = get_images_and_labels(images_dir)
        if not os.path.exists(performance):
            continue
        file_content = open(performance).read()
        accuracy, predicted_dir_labels = json.loads(file_content)
        accuracies.append(accuracy)
        true_labels.extend(dir_true_labels)
        predicted_labels.extend(predicted_dir_labels)
    return accuracies, true_labels, predicted_labels


def print_all_accuracies():
    accuracies, true_labels, predicted_labels = read_performance_txts()
    print('total accuracy', np.mean(accuracies))
    class_names = constants.cls2name.values()
    labels_by_class_dic = {cls: [] for cls in class_names}
    for idx, true_label in enumerate(true_labels):
        predicted_label = predicted_labels[idx]
        labels_by_class_dic[true_label].append(true_label == predicted_label)
    for cls, labels_bool_list in labels_by_class_dic.items():
        print(cls, np.mean(labels_bool_list))


if __name__ == '__main__':
    print_all_accuracies()
