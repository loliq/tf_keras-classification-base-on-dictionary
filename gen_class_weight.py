import os
import  math
import  numpy as np
import argparse

parser = argparse.ArgumentParser()
# Add argument
parser.add_argument('--label_map_path', required=False, default= "", help='path_to_labelmap')
parser.add_argument('--folder_path', required=False,default="", help='path train_dataset')


def cal_label_dict(label_map_path, folder_path):
    """

    :param label_map_path: label_map的path
    :param folder_path:  分类文件夹在的path
    :return:  dict = {key: label, value: num_example}
    """

def create_class_weight(labels_dict, mu = 0.15):
    """

    :param labels_dict:
    :param mu:
    :return:
    """
    total = np.sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu * total / float (labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    return  score


def romanToInt(s):
    d = {'I': 1, 'IV': 3, 'V': 5, 'IX': 8, 'X': 10, 'XL': 30, 'L': 50, 'XC': 80, 'C': 100, 'CD': 300, 'D': 500,
         'CM': 800, 'M': 1000}
    sum_num = 0
    i = 0
    while i < len(s):
        if s[i:i + 2] in d:
            sum_num += d[s[i:i + 2]]
            i += 2
        else:
            sum_num += d[s[i]]
            i += 1
    return sum_num
if __name__ == '__main__':
    romanToInt("IV")