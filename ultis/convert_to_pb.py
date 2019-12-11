#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/11 10:32
# @Author  : LLL
# @Site    : 
# @File    : convert_to_pb.py
# @Software: PyCharm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_weight_path',
                    default=r"E:\01-jupyter\08-tf2.0\05-easy_eager_classification\logs\ep003-loss0.066-val_loss0.068-val_acc0.445.h5",
                    required=False, help='path of input h5 file')
parser.add_argument('--input_config_path',
                    default=r"E:\01-jupyter\08-tf2.0\05-easy_eager_classification\logs\ DenseNet_lighter_config.json",
                    required=False, help='path of input model config file(.json file)')
parser.add_argument('--out_pb_path',
                    default=r"E:\01-jupyter\08-tf2.0\05-easy_eager_classification\logs\tmp.pb",
                    required=False, help='path of output pb file (.pbfile)')
parser.add_argument('--out_nodeName_path',
                    default=r"E:\01-jupyter\08-tf2.0\05-easy_eager_classification\logs\tmp.txt",
                    required=False, help='path of model node msg(.txt file)')

args = parser.parse_args()

import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.util import compat


# 转成pb
def keras_model_to_frozen_graph(model_config, model_weight, pb_file_path):
    """
    convert keras h5 model file to frozen graph(.pb file)
    :param model_config:  模型的
    :param model_weight:
    :param pb_file_path:
    :return:
    """
    import tensorflow as tf
    from tensorflow.python.framework import graph_io

    def freeze_graph(graph, session, output_node_names, model_name):
        with graph.as_default():
            tf.keras.backend.set_learning_phase(0) # this line most important
            graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
            graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output_node_names)
            graph_io.write_graph(graphdef_frozen, os.path.dirname(pb_file_path), os.path.basename(pb_file_path), as_text=False)
    with open(model_config) as json_file:
        json_config = json_file.read()
    tf.keras.backend.set_learning_phase(0) # this line most important
    model = keras.models.model_from_json(json_config)
    model.load_weights(model_weight)
    session = tf.keras.backend.get_session()
    freeze_graph(session.graph, session, [out.op.name for out in model.outputs], model_weight)

def write_node_name(model_path,out_nodemsg_path, print_all=False):
    """
     print input and output name
    :param model_path: path of pb file
    :return:
    """
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        with open(out_nodemsg_path, 'w') as f:
            if print_all:
                for i, n in enumerate(graph_def.node):
                    f.write("Name of the node - %s" % n.name)
                    f.write('\n')
            else:
                f.write("input node name :{}".format(graph_def.node[0].name))
                f.write('\n')
                f.write("output node name : {}".format(graph_def.node[-1].name))
                f.write('\n')
                print("input node name :{}".format(graph_def.node[0].name))
                print("output node name : {}".format(graph_def.node[-1].name))
    #         # 打印结点

if __name__ == '__main__':
    keras_model_to_frozen_graph(model_config=args.input_config_path,
                                model_weight=args.input_weight_path,
                                pb_file_path=args.out_pb_path)
    write_node_name(model_path=args.out_pb_path, out_nodemsg_path=args.out_nodeName_path)

