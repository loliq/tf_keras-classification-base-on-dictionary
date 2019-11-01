'''
- 需要的tensorflow 版本:tf.1.14
- 组织dataset 的输入map
- 对数据的处理，以及对dataset 的组织
- 对图像大小的处理方式
'''

import os
import numpy as np
import json
import glob
import PIL
from PIL import  Image
import matplotlib.pyplot as plt
import tensorflow as tf
tf.enable_eager_execution()

def compose_file_label(input_path, label_path):
    """
    用tf.eager()模式，组织tf.dataset
    - Input：
    > 1. 存有多个分类文件夹的路径，不同文件夹是不同类别的图片
    > 2. 路径中需要包含label 映射的label_map.txt(以字典的形式组织)：key 为文件夹名称，value为label
    > 3. 格式为 {"OK":0,"NG":1}
    - Output:
    > 1. image
    > 2. label: one_hot label(对应的compile loss 为 CategoricalCrossentropy) or interger-label(对应的loss 为 SparseCategoricalCrossentropy)

    :param input_path: 分类文件夹的路径
    :param label_path: la
    :return:
    """
    folders = [ i for i in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, i))]  #判断是否为
    with open(label_path, 'r') as  file:
        js = file.read()
        label_map = json.loads(js)
    file_names = []
    labels = []
    for folder in folders:
        full_folder = os.path.join(input_path, folder)
        sub_labels = [label_map[folder]] * len(os.listdir(full_folder))
        sub_folder_names = glob.glob(os.path.join(full_folder,"*"))
        labels += sub_labels
        file_names += sub_folder_names
    filenames_tensor = tf.constant(file_names)
    labels_tensor = tf.constant(labels)
    return filenames_tensor, labels_tensor

def _read_py_function(filename, label):
    """
    由于dataset中要用的函数需要为tf的函数，这边的作用是使用tf以外的python库文件
    :param filename: 文件名的Tensor形式
    :param label:
    :return:
    """
    decode_filename = filename.numpy().decode() # 将tf.eager_tensor转成numpy 再解码
    image_decoded = Image.open(decode_filename)
    image_decoded = np.array(image_decoded.convert("RGB")) # 转成3通道图像
    return image_decoded, label

def _preprocess_function(image_decoded, label,class_num=2,is_training=True,resize_shape=[224, 224, 3]):
    """
    这边写预处理函数
    :param image_decoded: 已经读取的文件
    :param label:
    :param class_num
    :return:
    """
    # TODO
    # reshap image and preprocess image
    tf_image = tf.reshape(image_decoded, resize_shape)
    if is_training:
        tf_image = tf.image.random_flip_left_right(tf_image)
        tf_image = tf.image.random_contrast(tf_image, 0.8, 1.2)
    tf_image = preprocess_image(tf_image)
    tf_label = tf.one_hot(label, class_num)
    return tf_image, tf_label

def preprocess_image(image):
    """
    图像预处理
    :param image:  输入图像
    :return:  预处理后的图像
    """
    image = tf.cast(image, tf.float32)
    image = image / 255.
    image -= 0.5
    image *= 2
    return image


def make_dataset_from_filenames(input_path,label_path,class_num=2,batch_size=1,is_training=True,resize_shape=[224, 224, 3],shuffle=True,shuffle_size=6000):
    """

    :param input_path: 分类文件夹所在的路径
    :param label_path: 标签label_map.txt 所在的路径
    :param batch_size:
    :param resize_shape:
    :param shuffle: 是否打乱数据
    :param shuffle_size:
    :return: dataset
    """
    filenames_tensor, labels_tensor = compose_file_label(input_path, label_path)
    dataset = tf.data.Dataset.from_tensor_slices((filenames_tensor, labels_tensor))
    dataset = dataset.map(
        lambda filename, label: tuple(tf.py_function(
            _read_py_function, [filename, label], [tf.uint8, label.dtype])))  # py_func 不能用于eager_tensor
    dataset = dataset.map(lambda image, label: _preprocess_function(image, label, class_num, is_training, resize_shape))
    dataset = dataset.batch(batch_size)
    if shuffle:
        dataset = dataset.shuffle(shuffle_size)
    return dataset

def fixed_ratio_resize(image, input_shape):
    """
    将输入图像按照长宽比不变的条件调整成网络所需要的图像大小. 不足的地方填0
    :param image: 输入的image， 由PIL.Image.open 读取
    :param input_shape: 网络固定的输入大小
    :return: reshape的图像大小
    """
    # 原始的图像的大小
    raw_w, raw_h = image.size
    # 网络的输入的大小
    input_w, input_h = input_shape
    ratio = min(input_w / raw_w, input_h / raw_h)
    new_w = int(raw_w * ratio)
    new_h = int(raw_h * ratio)
    dx = (input_w - new_w) // 2
    dy = (input_h - new_h) // 2
    image_data = 0
    # 关于为啥是128? 中心化后为0？
    # 图像长宽比不变 resize成正确的大小
    image = image.resize((new_w, new_h), Image.BICUBIC)
    new_image = Image.new('RGB', (input_w, input_h), (128, 128, 128))  # 三个通道都填128
    new_image.paste(image, (dx, dy))  #   # 图片在正中心,即若 dy = 50,则上下各填充50个像素

    return new_image

