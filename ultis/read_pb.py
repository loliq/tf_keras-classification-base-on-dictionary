import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.python.util import compat
from tensorflow.keras import backend as K
import numpy as np
import cv2

def read_image(filename, resize_height = 224, resize_width = 224):
    bgr_image = cv2.imread(filename)
    if len(bgr_image.shape) == 2:  # 若是灰度图则转为三通道
        print("Warning:gray image", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    if resize_height > 0 and resize_width > 0:
        rgb_image = cv2.resize(rgb_image, (resize_width, resize_height))
    rgb_image = np.asanyarray(rgb_image)
    return rgb_image
def read_preprocess(file_path):
    """
    预处理图像
    :param file_path:
    :return:
    """
    ori_image= read_image(file_path)
    image = tf.cast(ori_image, tf.float32)
    image = image / 255.
    image -= 0.5
    image *= 2
    image = tf.expand_dims(image,0)
    return image

def create_graph_pb(model_path, file_path):
    """

    :param model_path:  pb模型的路径
    :param file_path:  图片路径
    :return:
    """
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        # 打印结点
#         for i,n in enumerate(graph_def.node):
#             print("Name of the node - %s" % n.name)
    init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        #TODO REDEFINE THE INPUT TENSOR IF NESSASARY
        # 定义输入的张量名称,对应网络结构的输入张量
        # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
        input_image_tensor = sess.graph.get_tensor_by_name("input_1_1:0")
        output_tensor_name = sess.graph.get_tensor_by_name("fc1_1/Sigmoid:0")
        input_imge = read_preprocess(file_path)
        input_imge = sess.run(input_imge)

        score = sess.run(output_tensor_name, feed_dict={input_image_tensor: input_imge
                                                      })
        print("score:{}".format(score))
        class_id = tf.argmax(score, 1)
        print(class_id)
    #         print( "pre class is :{}".format(labels[sess.run(class_id)]))