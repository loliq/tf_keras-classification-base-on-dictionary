import tensorflow as tf
import argparse
import tensorflow as tf
from tensorflow import keras
import glob
import os
import numpy as np
import json
from ultis.dataset import make_dataset_tfrecord, anti_process
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
tf.enable_eager_execution()


parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', required=False, default='logs/error_dir')
parser.add_argument('--model_config', required=False,
                    default=r'logs\light_densenet\二分类增扩\DenseNet_lighter_config.json',
                    help='path of model archtecture config')
parser.add_argument('--model_weight', required=False,
                    default=r'logs\light_densenet\二分类增扩\ep230-loss0.029-val_loss0.059-val_acc0.876.h5',
                    help='path of model weight')
parser.add_argument('--class_num', required=False,
                    default=2,
                    help='class num')
parser.add_argument('--record_folder', required=False,
                    default=r'E:\01-VTC\01-伯恩\训练用图\1231暗场\record_file\01-binary增扩\val',
                    help='path of model weight')
args = parser.parse_args()

if __name__ == '__main__':
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    with open(args.model_config) as json_file:
        json_config = json_file.read()
    model = tf.keras.models.model_from_json(json_config)
    model.load_weights(args.model_weight)
    val_file_names = glob.glob(os.path.join(args.record_folder, '*'))
    val_dataset = make_dataset_tfrecord(filenames=val_file_names,
                                        batchsize= 4,
                                        is_training=False,
                                        classes_num=args.class_num,
                                        resize_shape=[224,224, -1])
    total_image_num = 0
    global_error_index = 0
    for image_batch, label_batch in val_dataset:
        total_image_num += label_batch.shape[0]
        gt = np.argmax(label_batch.numpy(), axis=1)
        # print("label is {}".format(gt))
        predict = np.argmax(model.predict(image_batch), axis=1)
        # print("predict is {}".format(predict))
        for index, [y_true, y_predict] in enumerate(zip(gt, predict)):

            if y_predict != y_true:
                reconstruct_image = anti_process(image_batch[index,...])
                fig = plt.figure(figsize=[8, 6])
                plt.title('label is {0}, pre is {1}'.format(y_true, y_predict))
                plt.imshow(np.squeeze(reconstruct_image.numpy()), cmap='gray')
                plt.savefig(args.out_dir + "/{:02d}.png".format(global_error_index))
                plt.close()
                global_error_index += 1

    print("total_image num is {}".format(total_image_num))
