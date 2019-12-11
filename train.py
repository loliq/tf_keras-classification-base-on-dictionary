import tensorflow as tf
import argparse

tf.enable_eager_execution()
import os
import json
from ultis.classification_model import cls_model
import importlib
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', required=False,
                    default=r'configs\light_densenet.py',
                    help='path of original dataset, which has train folder and val folder')
args = parser.parse_args()

def unpack_module(module_path):
    module_name, _ = os.path.splitext(module_path)
    dir_name = os.path.dirname(module_name)
    base_name = os.path.basename(module_name)
    module_name = dir_name + '.' + base_name
    return module_name

# FLAGS = flags.FLAGS
#
def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == '__main__':
    mod_config = importlib.import_module(unpack_module(args.config_path), __package__)
    check_dir(mod_config.model['work_dir'])
    model_object = cls_model(mod_config.model)
    model_object.train()
