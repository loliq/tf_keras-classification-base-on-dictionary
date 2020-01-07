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
import glob
from ultis.dataset import make_dataset_from_filenames, preprocess_image, make_dataset_tfrecord, anti_process
from ultis.losses_and_metrics import get_loss_obj
from PIL import Image
import tensorflow as tf
from tensorflow.python.framework import graph_io
import json
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

tf.enable_eager_execution()

# 设置中文字体和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.figsize'] = (24.0, 16.0)


"""
为了序列化方便，现在开始统一使用 save_weight 存 + json 文件存结构
"""

class cls_model(object):
    def __init__(self, config):
        self.config = config
        self._config_train()

    def _get_data(self):
        """
        获取训练数据
        :return:
        """
        train_file_names = glob.glob(os.path.join(self.config['data']['train']['image_dir'], '*'))
        print(train_file_names)
        val_file_names = glob.glob(os.path.join(self.config['data']['val']['image_dir'], '*'))
        print(val_file_names)
        # 对于tf.reshape。第三维用计算的方式填补，否则可能会报错
        train_dataset = make_dataset_tfrecord(filenames=train_file_names,
                                              batchsize=self.config['train_config']['batch_size'],
                                              is_training=True,
                                              classes_num=self.config['model_config']['classes'],
                                              resize_shape=[self.config['model_config']['input_shape'][0],
                                              self.config['model_config']['input_shape'][1], -1])
        val_dataset = make_dataset_tfrecord(filenames=val_file_names,
                                            batchsize=self.config['train_config']['batch_size'],
                                            is_training=False,
                                            classes_num=self.config['model_config']['classes'],
                                            resize_shape=[self.config['model_config']['input_shape'][0],
                                            self.config['model_config']['input_shape'][1], -1])

        self.label_map = self._get_label_map(self.config['data']['train']['label_path'])
        return train_dataset, val_dataset

    def _get_model_archtecture(self):
        """
        通过配置文件得到网络的结构
        :return:
        """
        if self.config['type'] == 'DenseNet':
            from models import DenseNet
            model_object = DenseNet.DenseNet(self.config['model_config'])
        if self.config['type'] == 'ResNet':
            from models import ResNet
            model_object = ResNet.ResNet(self.config['model_config'])
        if self.config['type'] == 'MobilenetV2':
            from models import MobileNet
            model_object = MobileNet.mobilenetV2(self.config['model_config'])

        self.model = model_object.constuct_model()

    def _get_optimizer(self):
        optimizer_name = self.config['train_config']['optimizer']['type']
        if optimizer_name == 'RMSprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.config['train_config']['base_lr'])

        return optimizer


    def _config_train(self):
        """
        设置训练参数
        :return:
        """
        # TODO 设置call_back
        check_point = keras.callbacks.ModelCheckpoint(self.config['work_dir'] +
                                                           '/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-val_acc{val_categorical_accuracy:.3f}.h5',
                                                           monitor='val_categorical_accuracy',
                                                           mode='auto', save_weights_only=True, save_best_only=True)
        lr_decay = keras.callbacks.LearningRateScheduler(self._learning_rate_schedule)
        logger = keras.callbacks.TensorBoard(self.config['work_dir'])
        self.call_backs = [check_point, lr_decay, logger]

        # 读入模型, 如果有预训练设置
        if self.config['pretrained']:
            # 读取模型结构
            with open(self.config['pretrained_config']['model_weight_path']) as json_file:
                json_config = json_file.read()
            self.model = keras.models.model_from_json(json_config)
            # 读取模型的权重
            self.model.load_weights(self.config['pretrained_config']['model_weight_path'])
        else:
            self._get_model_archtecture()
        # compile 设置
        loss_object = get_loss_obj(self.config['train_config']['loss'])
        optimizer = self._get_optimizer()
        self.model.compile(optimizer=optimizer,
                           loss=loss_object,
                           metrics=['categorical_accuracy'])

    def _learning_rate_schedule(self, epoch):
        """

        :param epoch:
        :return:
        """
        if epoch < self.config['train_config']['lr_config']['warmdown_iters']:
            return self.config['train_config']['base_lr']
        else:
            lr = self.config['train_config']['base_lr'] \
                 * np.math.pow(self.config['train_config']['lr_config']['warmdown_ratio'],
                               np.floor((epoch - self.config['train_config']['lr_config']['warmdown_period']) /
                                        self.config['train_config']['lr_config']['warmdown_period']))
            return lr

    def train(self):
        # 存下网络配置
        json_config = self.model.to_json()
        with open(self.config['work_dir'] + '/' + self.config['model_config']['type'] + '_config.json', 'w') as json_file:
            json_file.write(json_config)

        train_dataset, val_dataset = self._get_data()
        self.history = self.model.fit(train_dataset,
                                      initial_epoch=0,
                                      epochs=self.config['train_config']['total_epoches'],
                                      callbacks=self.call_backs,
                                      validation_data=val_dataset
                                      )
        self.model.save_weights(self.config['work_dir'] + "/final_epoch.h5")
        self._plot_train_msg(self.config['work_dir'])

    def _get_class_num(self):
        """

        :return: class_num_dict 类别 0 -n 的类别数量
        """
        # class_num_dict 是
        class_num_dict = {}
        # label_map是文件夹名称对应的缺陷的编号
        for folder in os.listdir(self.config['data']['train']['image_dir']):
            if os.path.isdir(os.path.join(self.config['data']['train']['image_dir'], folder)):
                if self.label_map[folder] in class_num_dict:
                    class_num_dict[self.label_map[folder]] +=\
                        len(os.listdir(os.path.join(self.config['data']['train']['image_dir'], folder)))
                else:
                    class_num_dict[self.label_map[folder]]\
                        = len(os.listdir(os.path.join(self.config['data']['train']['image_dir'], folder)))
        print(class_num_dict)
        keys = list(class_num_dict.keys())
        keys.sort()  # 排序
        class_nums = [class_num_dict[i] for i in keys]
        print(class_nums)
        return class_nums
    def _get_label_map(self, label_path):
        with open(label_path, 'r') as file:
            js = file.read()
            label_map = json.loads(js)
        return label_map

    def _plot_train_msg(self, save_path):
        acc = self.history.history['categorical_accuracy']
        val_acc = self.history.history['val_categorical_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.ylim([0.8, 1])
        plt.plot([0, 0],
                 plt.ylim(), label='Start Train')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.ylim([0, 1.0])
        plt.plot([0, 0],
                 plt.ylim(), label='Start Train')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.savefig(save_path + "/train_msg.jpg")
        plt.show()

    def test_image(self, image_path):
        """

        :param image_path: single image
        :return: ypred, 网络输出的置信度值
        """
        #TODO image_preprocess
        image = np.array(Image.open(image_path))
        image = preprocess_image(image)
        image = tf.expand_dims(image, 0)
        y_pred = self.model.predict(image)
        class_index = np.argmax(y_pred, axis=1)
        return y_pred

    # TODO 测试混淆矩阵
    def evaluate_dataset(self, label_dict, dataset_name):
        """
        评估数据集
        - 存储混淆矩阵
        - 保存分错的图像
        :param label_dict: 每一个标签对应的分类名称
        :param dataset_name:
        :return:
        """
        # 读取最优秀的模型
        gt = []
        pred = []
        for index, [image, label] in tqdm(enumerate(self.val_dataset)):
            #     result = [batchsize, class_num] if multi-classification
            #     result = [batchsize, 1] if binary_classification
            y_pred = self.model.predict(image)
            y_pred = np.argmax(y_pred, axis=1)
            pred += pred.tolist()
            y_true = np.argmax(label.numpy(), 1).tolist()
            gt += y_true
            if y_pred[0] != y_true[0]:
                raw_image = tf.cast(anti_process(image), tf.uint8)
                plt.figure()
                plt.imshow(raw_image[0, ...])
                plt.title("label is{0}, pred is {1}".format(label_dict[y_true[0]], label_dict[y_pred[0]]))
                save_path = os.path.join(self.config['work_dir'], 'wrong classification')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plt.savefig(save_path + '/{}.png'.format(index))
        label_name = [str(i) for i in range(self.config['model_config']['class_num'])]
        self._plot_confusion_matrix(label_name, save_path=self.config['work_dir'], y_true=gt, y_pred=y_pred)



    def save_to_pb(self, pb_name):
        def freeze_graph(graph, session, output_node_names, model_name):
            with graph.as_default():
                graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
                graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output_node_names)
                graph_io.write_graph(graphdef_frozen, self.config['work_dir'], os.path.basename(model_name) + ".pb", as_text=False)

        session = tf.keras.backend.get_session()
        freeze_graph(session.graph, session, [out.op.name for out in self.model.outputs], pb_name)
        print("freezing end")


    def _plot_confusion_matrix(self,labels, save_path, y_true, y_pred,fontsize=10, title="Confusion Matrix"):
        """
        输入标签名称，输出混淆矩阵的图
        :param labels:
        :param y_true:
        :param y_pred:
        :param fontsize:
        :param title:
        :return:
        """
        tick_marks = np.array(range(len(y_true) + len(y_pred))) + 0.5
        cm = confusion_matrix(y_true, y_pred)
        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(8,6), dpi=120)
        ind_array = np.arange(len(labels))
        x, y = np.meshgrid(ind_array, ind_array)
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm[y_val][x_val]
            cp = cm_normalized[y_val][x_val]
            if (c > 0.01):
                plt.text(x_val, y_val, "%d" %(c,), color='red', fontsize=fontsize, va='top', ha='center')
                plt.text(x_val, y_val, "%0.4f %s" % (cp,'%'), color='red', fontsize=fontsize, va='bottom', ha='center')

        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=90)
        plt.yticks(xlocations, labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # show confusion matrix
        plt.show()
        # save_confusion matrix
        plt.savefig(save_path + "/train_line.jpg")


