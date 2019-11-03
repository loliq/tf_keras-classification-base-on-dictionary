#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
代码说明：
1. model 保存为了格式通用，统一使用 `model.json` + 模型权重`model_weight.h5` 保存模型
2. 为了尽量不破坏源码， 在使用自定义的`loss`和`metrics`的时候，不将函数加入后台的keras源码，而是在需要定义compile的时候导入对应的函数，因此使用的时候要注意
3. label 的格式为了适用focal_loss 统一使用 one_hoe_label
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from models import DenseNet
from ultis.dataset import make_dataset_from_filenames, preprocess_image
from PIL import Image
from tensorflow.python.util import compat
from tensorflow.keras.models import Model, load_model
tf.enable_eager_execution()
import tensorflow as tf
from tensorflow.python.framework import graph_io
import json
from losses_and_metrics import multi_category_focal_loss_class_num
"""
为了序列化方便，现在开始统一使用 save_weight 存 + json 文件存结构
"""

class cls_model(object):
    def __init__(self, config, input_shape):
        self.config = config
        self.input_shape = input_shape
        class_nums = self._get_class_num()
        # 只保存权重
        self.check_point = keras.callbacks.ModelCheckpoint(self.config.logdir + '/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',monitor='val_categorical_accuracy',
                                                           mode='auto', save_weights_only=True, save_best_only=True)
        self.lr_decay = keras.callbacks.LearningRateScheduler(self._learning_rate_schedule)
        self.logger = keras.callbacks.TensorBoard(self.config.logdir)
        self.call_backs = [self.check_point, self.lr_decay, self.logger]
        # TODO 创建dataset
        self.train_dataset = make_dataset_from_filenames(input_path=self.config.train_folder_path,
                                                         label_path=self.config.label_path,
                                                         batch_size=self.config.batch_size,
                                                         class_num=self.config.class_num,
                                                         resize_shape=input_shape,
                                                         is_training=True,
                                                         shuffle=True)
        if self.config.val_folder_path is not None:
            self.val_dataset = make_dataset_from_filenames(input_path=self.config.val_folder_path,
                                                           label_path=self.config.label_path,
                                                           batch_size=self.config.batch_size,
                                                           class_num=self.config.class_num,
                                                           resize_shape=input_shape,
                                                           is_training=False,
                                                           shuffle=False)
        if self.config.load_pretrained:
            # 读取模型结构
            with open(self.config.model_config_path) as json_file:
                json_config = json_file.read()
            self.model = keras.models.model_from_json(json_config)
            # 读取模型的权重
            self.model.load_weights(self.config.model_path)
        else:
            self.model = DenseNet.DenseNet_lighter(classes=self.config.class_num, nb_filter=32, growth_rate=8,
                                                   dropout_rate=0.3, reduction=0.5)
            # 存模型文件
            json_config = self.model.to_json()
            with open(self.config.logdir + '/model_config.json', 'w') as json_file:
                json_file.write(json_config)

    def _learning_rate_schedule(self,epoch):
        if epoch < 200:
            return self.config.base_lr
        else:
            return 0.001 * np.math.pow(0.9, np.floor((epoch - 200) / 4))
    def train(self):
        # TODO 配置网络训练参数
        self.model.compile(optimizer=keras.optimizers.Adam(self.config.base_lr),
                           loss='categorical_crossentropy',
                           metrics=['categorical_accuracy'])
        if self.config.val_folder_path is not None:
            self.history = self.model.fit(self.train_dataset,
                                          initial_epoch=0,
                                          epochs=self.config.first_epochs,
                                          callbacks=self.call_backs,
                                          validation_data=self.val_dataset
                                          )
    def _get_class_num(self):
        """

        :return: class_num_dict 类别 0 -n 的类别数量
        """
        # class_num_dict 是
        class_num_dict = {}
        label_path = self.config.label_path
        with open(label_path, 'r') as  file:
            js = file.read()
            label_map = json.loads(js)
        # label_map是文件夹名称对应的缺陷的编号
        for folder in os.listdir(self.config.train_folder_path):
            if os.path.isdir(os.path.join(self.config.train_folder_path, folder)):
                if label_map[folder] in class_num_dict:
                    class_num_dict[label_map[folder]] += len(os.listdir(os.path.join(self.config.train_folder_path, folder)))
                else:
                    class_num_dict[label_map[folder]] = len(os.listdir(os.path.join(self.config.train_folder_path, folder)))
        print(class_num_dict)
        keys = list(class_num_dict.keys())
        keys.sort()  # 排序
        class_nums = [class_num_dict[i] for i in keys]
        print(class_nums)
        return class_nums

    def test_image(self,image_path):
        """

        :param image_path:
        :return:
        """
        #TODO image_preprocess
        image = np.array(Image.open(image_path))
        image = preprocess_image(image)
        image = tf.expand_dims(image, 0)
        result = self.model.predict()
        return

    def save_to_pb(self, pb_name):
        def freeze_graph(graph, session, output_node_names, model_name):
            with graph.as_default():
                graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
                graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output_node_names)
                graph_io.write_graph(graphdef_frozen, self.config.logdir, os.path.basename(model_name) + ".pb", as_text=False)

        session = tf.keras.backend.get_session()
        freeze_graph(session.graph, session, [out.op.name for out in self.model.outputs], pb_name)
        print("freezing end")

