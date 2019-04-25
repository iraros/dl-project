from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, 'rb') as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = 'file_reader'
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith('.png'):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name='png_reader')
    elif file_name.endswith('.gif'):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name='gif_reader'))
    elif file_name.endswith('.bmp'):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    elif file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name='jpeg_reader')
    else:
        raise ValueError('Unsupported image type for file "{}"'.format(file_name))
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


input_layer = 'Placeholder'
output_layer = 'final_result'
input_name = 'import/' + input_layer
output_name = 'import/' + output_layer


def get_images_labels_scores(file_names, model_path, labels_path):
    labels = load_labels(labels_path)

    graph = load_graph(model_path)
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    image_scores_dicts = []
    image_num = len(file_names)
    for idx, file_name in enumerate(file_names):
        print('classifying: {} / {}, - {}, should be: {}'.format(idx, image_num, os.path.basename(file_name),
                                                                 os.path.basename(os.path.dirname(file_name))))
        image_scores = apply_graph(file_name, graph, input_operation, output_operation)
        image_scores_dict = dict(zip(labels, image_scores))
        image_scores_dicts.append(image_scores_dict)

    return image_scores_dicts


def apply_graph(file_name, graph, input_operation, output_operation):
    t = read_tensor_from_image_file(file_name)
    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        })
    return np.squeeze(results)


def get_labels_string_for_one_image(image, model_path, labels_path):
    labels_result = get_images_labels_scores([image], model_path, labels_path)
    labels_sorted = sorted(labels_result[0].items(), key=lambda kv: -kv[1])
    labels_rounded = [(str(round(item[1], 4)), item[0]) for item in labels_sorted]
    labels_string = '\n'.join([' '.join(label) for label in labels_rounded])
    return labels_string


def classify_images(file_names, model_path, labels_path):
    labels_results = get_images_labels_scores(file_names, model_path, labels_path)
    max_labels = [max(labels_result.keys(), key=lambda k: labels_result[k]) for labels_result in labels_results]
    max_labels = [max_label.replace(' ', '_') for max_label in max_labels]
    return max_labels
