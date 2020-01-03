'''
- 需要的tensorflow 版本:tf.1.14
- 组织dataset 的输入map
- 对数据的处理，以及对dataset 的组织
- 对图像大小的处理方式: 使用 fixed_ratio_resize 改变图像的大小
- 尚未修复的bug ,不知道为什么 model.predict的时候，用make_dataset_from_filenames的时候会卡住， 总之先不管了。。。。
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
    folders = [ i for i in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, i))]  #判断是否为文件夹
    with open(label_path, 'r') as  file:
        js = file.read()
        label_map = json.loads(js)
    file_names = []
    labels = []
    for folder in folders:
        full_folder = os.path.join(input_path, folder)
        sub_labels = [label_map[folder]] * len(os.listdir(full_folder))  # 子文件夹的label为该文件夹的文件个数 * 文件夹名对应的字典值
        sub_folder_names = glob.glob(os.path.join(full_folder, "*")) # 文件名为其路径
        labels += sub_labels
        file_names += sub_folder_names
    # print(labels)
    filenames_tensor = tf.constant(file_names)  # 将文件名和label 转成Tensor形式
    labels_tensor = tf.constant(labels)
    return filenames_tensor, labels_tensor

def _read_py_function(filename, label, resize_shape):
    """
    由于dataset中要用的函数需要为tf的函数，这边的作用是使用tf以外的python库文件
    :param filename: 文件名的Tensor形式
    :param label:
    :return:
    """
    decode_filename = filename.numpy().decode() # 将tf.eager_tensor转成numpy 再解码
    image_decoded = Image.open(decode_filename)
    # TODO 关于如何reshape image
    image_decoded = image_decoded.convert("RGB")# 转成3通道图像
    image_decoded = fixed_ratio_resize(image_decoded, (resize_shape[0], resize_shape[1]))
    image_decoded = np.array(image_decoded) # 转成array的形式
    return image_decoded, label

def _preprocess_function(image_decoded, label,class_num=2,is_training=True,resize_shape=[224, 224, 3]):
    """
    这边写预处理函数
    :param image_decoded: 已经读取的图像
    :param label:
    :param class_num
    :return:
    """
    tf_image = image_decoded
    # tf_image = tf.reshape(tf_image, [224, 224, 3])
    if is_training:
        tf_image = tf.image.random_flip_left_right(tf_image)
        # tf_image = tf.image.random_contrast(tf_image, 0.8, 1.2)
    # TODO 如何预处理图像
    tf_image = tf.cast(tf_image, tf.float32)
    tf_image = tf_image / 255.
    tf_image -= 0.5
    tf_image *= 2
    # tf_image = preprocess_image(tf_image)
    # TODO 将label转成one_hot形式
    tf_label = tf.one_hot(label, class_num, 1, 0)
    return tf_image, tf_label

# TODO 定义图像预处理函数

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

# TODO 将预处理后的图像转回原图
def anti_process(image):
    """
    针对预处理的反预处理，用于从record_dataset中显示图像
    :param image:
    :return:
    """
    image = image / 2.
    image += 0.5
    image *= 255.
    image = tf.cast(image, tf.uint8)
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

    # 这边的lambda 属于参数捕获进去，从dataset中捕获filename 和label输入到 _read_py_function
    dataset = dataset.map(
        lambda filename, label: tuple(tf.py_function(
            _read_py_function, [filename, label, resize_shape], [tf.uint8, tf.int32])))  # py_func 不能用于eager_tensor
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
    if input_h == raw_h and input_w == raw_w:
        return image
    else:
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


# TODO TF-Record

def parse_single_exmp(serialized_example,process_func=None,is_training=True, label_num=2,
                      resize_shape=None):
    """
    解析tf.record
    :param serialized_example:
    :param opposite: 是否将图片取反
    :return:
    """
    # 解序列化对象
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
    )
    tf_image = tf.io.decode_raw(features['image_raw'],tf.uint8)#获得图像原始的数据
    tf_label = tf.cast(features['label'], tf.int32)
    # TODO 图像大小不同的时候需要修改
    tf_image = tf.reshape(tf_image, resize_shape)  # 设置图像的维度
    if is_training:
        # TODO 这里做训练时候的数据增强
        tf_image = tf.image.random_flip_left_right(tf_image)
        # tf_image = tf.image.random_contrast(tf_image, 0.8, 1.2)
    tf_image = preprocess_image(tf_image)
    tf_label = tf.one_hot(tf_label, label_num, 1, 0)  #二分类只需要 0 和1
    return tf_image, tf_label

def make_dataset_tfrecord(filenames,batchsize=8, is_training = True, classes_num=2, resize_shape=[224,224,3]):
    dataset = tf.data.TFRecordDataset(filenames)
    # lambda x 取到dataset的serial_sample对象
    dataset = dataset.map(lambda x: parse_single_exmp(x, is_training=is_training, label_num=classes_num, resize_shape=resize_shape))
    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batchsize)
    return dataset

####  另外一种从文件夹中读取图片并且做标签
