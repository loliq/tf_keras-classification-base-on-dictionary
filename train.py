import tensorflow as tf
tf.enable_eager_execution()
import os
import json
from classification_model import cls_model

flags = tf.app.flags
flags.DEFINE_float('val_split', 0.1, 'validation rate')
flags.DEFINE_float('base_lr', 0.001, 'base_learning_rate')
flags.DEFINE_string("logdir", "logs", "Directory for storing checkpoint")
flags.DEFINE_string("train_folder_path", "E:/01-jupyter/11-VTC/01-BOEN盖板/01-Tensorflow框架/raw_image/cover_light/train_augmentation", "folder with classes path")
flags.DEFINE_string("val_folder_path", "E:/01-jupyter/11-VTC/01-BOEN盖板/01-Tensorflow框架/raw_image/cover_light/val", "folder with classes path")
flags.DEFINE_string("label_path", "E:/01-jupyter/11-VTC/01-BOEN盖板/01-Tensorflow框架/raw_image/label_map.txt", "label map path")
flags.DEFINE_integer("batch_size", 4, "batch_size")
flags.DEFINE_integer("class_num", 8 , "class number")
flags.DEFINE_boolean("is_training", True, "is_training")
flags.DEFINE_boolean("load_pretrained", False, "if load pretrained model")
flags.DEFINE_integer("first_epochs", 2, "epoches of first stage")
flags.DEFINE_string("model_config_path", "model_data/model_config.json", "pretrained_model")
flags.DEFINE_string("model_path", "model_data/pretrained.h5", "pretraind_weight_model")
# 关于类的比例可以根据这个修改
flags.DEFINE_string("class_weight_file", "model_data/class_weight")

FLAGS = flags.FLAGS
def check_dir():
    if not os.path.exists(FLAGS.logdir):
        os.mkdir(FLAGS.logdir)
def main(_):
    check_dir()
    print(tf.__version__)
    model = cls_model(FLAGS, [224,224,3])
    if FLAGS.is_training:
        model.train()


if __name__ == '__main__':
    tf.app.run()
