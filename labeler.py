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


model_base = 'datas/UrbanSound8K/output_graphs/'
labels_base = 'datas/UrbanSound8K/output_labels/'


def label_image(file_name, model_name, label_name):
    input_layer = 'Placeholder'
    output_layer = 'final_result'
    graph = load_graph(os.path.join(model_base, model_name + '.pb'))

    t = read_tensor_from_image_file(file_name)
    input_name = 'import/' + input_layer
    output_name = 'import/' + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)
    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        })
    results = np.squeeze(results)
    # top_k = results.argsort()[-5:][::-1]
    labels = load_labels(os.path.join(labels_base, label_name + '.txt'))
    return dict(zip(labels, results))


def get_labels_string(image, model_name, label_name):
    labels_result = label_image(image, model_name, label_name)
    labels_sorted = sorted(labels_result.items(), key=lambda kv: -kv[1])
    labels_rounded = [(str(round(item[1], 4)), item[0]) for item in labels_sorted]
    labels_string = '\n'.join([' '.join(label) for label in labels_rounded])
    return labels_string
