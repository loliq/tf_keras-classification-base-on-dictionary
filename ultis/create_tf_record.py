#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/3 14:04
# @Author  : LLL
# @Site    : 
# @File    : create_tf_record.py
# @Software: PyCharm

import tensorflow as tf
import json
import os
import glob
import random
from tqdm import tqdm
from  PIL import Image
import contextlib2
import numpy as np
import argparse
import cv2

parser = argparse.ArgumentParser()
# Add argument
parser.add_argument('--origin_dir', required=True, help='path of original dataset, which has train folder and val folder')
parser.add_argument('--out_dir', required=True, help='folder path to save tf_record file')
parser.add_argument('--label_map_path', required=True, help='label map of folders')
parser.add_argument('--resize_height', type=int, help='number of dataset', default=224)
parser.add_argument('--resize_width', type=int, help='number of dataset', default=224)
parser.add_argument('--resize_channel', type=int, help='number of dataset', default=1)
args = parser.parse_args()

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
    input_w, input_h, channels = input_shape
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
        if channels == 3:
            new_image = Image.new('RGB', (input_w, input_h), (128, 128, 128))  # 三个通道都填128
        if channels == 1:
            new_image = Image.new('L',(input_w, input_h), (128))
        else:
            new_image = None
            assert  new_image != None
        new_image.paste(image, (dx, dy))  #   # 图片在正中心,即若 dy = 50,则上下各填充50个像素

        return new_image


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value])) # if Value is not list,then add[]


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 生成实数型的属性
def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def load_label_map(label_map_path):
    with open(label_map_path, 'r') as file:
        js = file.read()
        label_map = json.loads(js)
        print(label_map)
    return label_map

def compose_label_filename(images_dir, label_map):
    """
    将文件夹中的图片的路径，按照file_name_path, label 的形式存下
    :param images_dir: 存放数据集的路径
    :param label_map: 每个文件夹对应的标签映射
    :return: 返回一个列表，列表里面是[(filename, label)] 的tuple
    """
    folders = [i for i in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, i))]
    label_filename = []
    for folder in folders:
        full_folder = os.path.join(images_dir, folder)
        #     sub_labels = [folder] * len(os.listdir(full_folder))  # 子文件夹的label为该文件夹的文件个数 * 文件夹名对应的字典值
        sub_folder_names = glob.glob(os.path.join(full_folder, "*"))  # 文件名为其路径
        for file_name in sub_folder_names:
            label_filename.append((file_name, label_map[folder]))
    return label_filename

def create_record_file(label_filenames,output_record_name, instances_per_shard,reshape_size = [224, 224, 3], shuffle=True):
    """

    :param label_filenames:
    :param output_record_dir:
    :param file_name:
    :param instances_per_shard:
    :param shuffle:
    :return:
    """
    # 打乱列表
    random.shuffle(label_filenames)
    num_example = len(label_filenames)
    num_train_shards = num_example // instances_per_shard
    for index in tqdm(range(num_example)):
        if index == 0:
            filename = (output_record_name + '-%.2d-of-%.2d' % (0, num_train_shards))
            print(filename)
            writer = tf.python_io.TFRecordWriter(filename)
        if index % instances_per_shard == 0 and index != 0:
            writer.close()
            tf_index = index / instances_per_shard
            filename = (output_record_name+'-%.2d-of-%.2d' % (tf_index, num_train_shards))
            print(filename)
            writer = tf.python_io.TFRecordWriter(filename)
        image_path = label_filenames[index][0]
        label = label_filenames[index][1]
        if not os.path.exists(image_path):
            print('Err:no image',image_path)
            continue
        image = Image.open(image_path)
        if reshape_size[2] == 3:
            image = image.convert("RGB")
        elif reshape_size[2] == 1:
            image = image.convert("L")
        else:
            image = None
            assert image != None
        image = fixed_ratio_resize(image, reshape_size)
        image_array = np.array(image)
        if len(image_array.shape) == 3:
            print(image_array.shape)
            print(label_filenames[index][0])
        image_raw = image_array.tostring()
        # TODO 此处加入要序列化的对象
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'label': _int64_feature(label)
        }))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    # TODO
    origin_dir1 = args.origin_dir
    result_dir = args.out_dir
    label_map_path = args.label_map_path
    train_regrex = result_dir + '/train/train-'
    val_regrex = result_dir + '/val/val-'
    dataSet_regrex = [train_regrex, val_regrex]
    # TODO
    reshape_size = [args.resize_height, args.resize_width, args.resize_channel]

    label_map = load_label_map(label_map_path)
    train_label_filenames = compose_label_filename(origin_dir1 + '/train_augmentation', label_map)
    val_label_filenames = compose_label_filename(origin_dir1 + '/val', label_map)
    # 计算类别分布比例
    label_distribution = {}
    for _, label in train_label_filenames:
        if str(label) in label_distribution:
            label_distribution[str(label)] += 1
        else:
            label_distribution[str(label)] = 1

    print(label_distribution)
    with open(os.path.dirname(result_dir) + '/train_class_num_distribution.txt', 'w') as f:
        f.write(str(label_distribution))

    for regrexName in dataSet_regrex:
        path = os.path.dirname(regrexName)
        if not os.path.exists(path):
            os.makedirs(path)
    with open(result_dir + "\class_num_distribution.txt", 'w') as f:
        for num in label_distribution:
            f.write(str(num) + ',')
    instances_per_shard = 500
    create_record_file(train_label_filenames, train_regrex, instances_per_shard, reshape_size)
    create_record_file(val_label_filenames, val_regrex, instances_per_shard, reshape_size)


