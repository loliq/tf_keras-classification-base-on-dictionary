# from utils import gen_data_txt,write_label_text,trans_dir,gen_data_txt_with_labels,segment_k_fold, compose_to_dataset
import os
import shutil
import argparse
import numpy as np
# Create ArgumentParser() object
from tqdm import tqdm
parser = argparse.ArgumentParser()
# Add argument
parser.add_argument('--data_path', required=False, default=r'H:\01-VTC\01-伯恩\训练用图\1108暗场\原图', help='path to original dataset')
parser.add_argument('--dst_path', required=False, default=r'H:\01-VTC\01-伯恩\训练用图\1108暗场\5_folds', help='path to segment dataset')
parser.add_argument('--num_folds', type=int, help='number of cross_validation folds', default=5)
args = parser.parse_args()


def trans_dir(oriDir):
    transformed_dir = re.sub(r'\\', '/', oriDir)
    return transformed_dir


def gen_data_txt(rootdir):
    """
    生成元数据集的索引
    :param rootdir:
    :return:
    """
    sub_dirs = [x[0] for x in os.walk(rootdir)]
    del sub_dirs[0]
    with open(os.path.join(rootdir,"data.txt"), 'w') as f:
        base_names = os.path.basename(sub_dirs[0])
        print(base_names)
        for sub_dir in sub_dirs:
            base_name = os.path.basename(sub_dir)
            list0 = os.listdir(sub_dir)
            for j in range(0, len(list0)):
                f.write(base_name + "/" + list0[j] + '\n')

def create_dir(src_path, data_path, delete = True):

    """
    判断文件夹是否存在，删除文件夹的内容
    :param dataPath:  数据集根目录
    :param labels:
    :param delete:
    :return:   ###注意os.mkdir会报错要用os.makedirs
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if delete is True:
        shutil.rmtree(data_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    labels = write_label_text(src_path, data_path)     # 写label文件
    dataset_labels = ['train','val', 'test']
    for dataset_label in dataset_labels:
        dir_path1 = os.path.join(data_path, dataset_label)
        for label in labels:
            dir_path = os.path.join(dir_path1, label)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)


#写label名称
def write_label_text(src_path, dst_path):
    """
    写标签数据
    :param src_path: 文件夹根目录，目录下为分类的文件夹
    :return:
    """
    labels = []
    for file in os.listdir(src_path):
        if os.path.isdir(os.path.join(src_path, file)):
            labels.append(file)
    if len(labels):
        with open(os.path.join(dst_path, 'label.txt'), 'w') as f:
            for label in labels:
                f.write(label + '\n')
    return labels


def separate_data(src_path, dst_path, trainPer=0.7, valPer =0.15):
    """
    复制数据并分成验证集,测试集和训练集.并写成txt文件
    :param src_path:
    :param dst_path:
    :param trainPer:
    :param valPer:
    :return:
    """
    data_path = os.path.join(src_path, 'data.txt')
    if os.path.isfile(data_path):
        with open(data_path,'r') as f:
            lines = f.readlines()
            file_index_array = np.random.permutation(len(lines)) #生成随机序列
            total_num = len(lines)
            train_num = int(np.floor(total_num * trainPer))
            val_num = int(np.floor(total_num * valPer))
            test_num = total_num - train_num - val_num
            train_index = file_index_array[0:train_num-1]
            val_index = file_index_array[train_num: total_num-test_num-1]
            test_index = file_index_array[total_num-test_num:total_num-1]
            dataset_indexes = [train_index, val_index, test_index]
            dataset_indexes = list(dataset_indexes)
            dataset_names = ['train', 'val', 'test']
            dataset_names = list(dataset_names)
            labels = write_label_text(src_path, dst_path)

            for i in range(3):
                print(dataset_names[i])
                dataset_path = os.path.join(dst_path, dataset_names[i])
                with open(os.path.join(dst_path,dataset_names[i] + '.txt'), 'w') as fileWriter:
                    for index in dataset_indexes[i]:
                        line = lines[index].strip('\n')
                        shutil.copy(os.path.join(src_path, line), os.path.join(dataset_path, line))
                        label_name = os.path.dirname(line)
                        #找出索引
                        label_index = labels.index(label_name)
                        fileWriter.write(line + " " + str(label_index) + '\n')

def gen_data_txt_with_labels(rootdir, fileName, label_file):
    """
    生成元数据集的索引
    :param rootdir:
    :return:
    """
    parent_dir = os.path.dirname(rootdir)
    label_list = []
    if os.path.exists(label_file):
        with open(label_file,'r') as f:
           label_list = f.readlines()
           label_list = [ label.strip('\n') for label in label_list ]


    class_folders = os.listdir(rootdir)
    class_paths = []
    for class_folder in class_folders:
        folder_path = os.path.join(rootdir,class_folder)
        if os.path.isdir(folder_path):
            class_paths.append(folder_path)

    with open(os.path.join(parent_dir, fileName),'w') as f:
        for class_folder in class_folders:
            folder_path = os.path.join(rootdir, class_folder)
            if os.path.isdir(folder_path):
                image_files = os.listdir(folder_path)
                for image_file in image_files:
                    f.write("{0}/{1} {2}\n".format(class_folder, image_file,
                                                 label_list.index(class_folder)))


def segment_k_fold(data_path, dst_path, num_fold):
    """
    :param data_path:
    :param num_fold:
    :return:
    """
    data_list = os.path.join(data_path, 'data.txt')
    with open(data_list, 'r') as f:
        lines = f.readlines()
        file_index_array = np.arange(0,len(lines))
        np.random.shuffle(file_index_array)  # 生成随机序列
        labels = write_label_text(data_path, dst_path)
        total_sample_num = len(file_index_array)
        per_fold = 1.0 / num_fold
        for fold_index in range(num_fold):
            fold_dir_name = "{0}_of_{1}".format(fold_index+1, num_fold)
            fold_path_name = os.path.join(dst_path, fold_dir_name)
            if not os.path.exists(fold_path_name):
                os.makedirs(fold_path_name)
                for label in labels:
                    os.makedirs(os.path.join(fold_path_name, label))

            data_start = 0
            data_end = 0
            if fold_index == (num_fold-1):
                data_start = int(np.floor((fold_index)*per_fold*total_sample_num))
                data_end = total_sample_num - 1
            else:
                data_start = int(np.floor(fold_index*per_fold*total_sample_num))
                data_end = int(np.floor((fold_index + 1)*per_fold*total_sample_num)-1)
            print("data start" + str(data_start))
            print("data end" + str(data_end))
            lines_array= file_index_array[data_start:data_end]
            print(lines_array)
            for line_index in lines_array:
                line = lines[line_index].strip('\n')
                shutil.copy(os.path.join(data_path, line), os.path.join(fold_path_name, line))

def compose_to_dataset(folder_pathes, dst_path):

    for folder in folder_pathes:
       subdirs =  os.listdir(folder)
       for subdir in subdirs:
           src_folder = os.path.join(folder, subdir)
           if os.path.isdir(src_folder):
               dst_folder = os.path.join(dst_path, os.path.basename(subdir))
               if not os.path.exists(dst_folder):
                   os.makedirs(dst_folder)
               for file in os.listdir(os.path.join(folder, subdir)):
                  shutil.copy(os.path.join(src_folder, file),
                              os.path.join(dst_folder, file))
               # shutil.copytree(folder, os.path.join(dst_folder))

if __name__ == '__main__':
    for root, dirs, files in os.walk(args.data_path):
        for dir in dirs:
            print("类别: {0} 共有{1}张图像".format(os.path.basename(dir), len(os.listdir(os.path.join(root, dir)))))
    print("------------------------------ file segment begin--------------------------------------")
    gen_data_txt(args.data_path)
    #
    label = write_label_text(args.data_path, args.dst_path)
    segment_k_fold(args.data_path, args.dst_path, args.num_folds)
    #
    for index in tqdm(range(args.num_folds)):
        val_name = "{0}_of_5".format(index + 1)
        nameList = ["{0}_of_5".format(i+1) for i in range(args.num_folds)]
        nameList.remove(val_name)
        nameDir = [os.path.join(args.dst_path, name) for name in nameList]
        dst_train = args.dst_path + '/{0}_of_5folds/train'.format(index + 1)
        dst_val = args.dst_path + '/{0}_of_5folds/val'.format(index + 1)
        compose_to_dataset(nameDir, dst_train)
        gen_data_txt_with_labels(dst_train,
                                 "train.txt",
                                 os.path.join(args.dst_path, 'label.txt'))

        shutil.copytree(os.path.join(args.dst_path, val_name),
                        dst_val)
        gen_data_txt_with_labels(dst_val,
                                 "val.txt",
                                 os.path.join(args.dst_path, 'label.txt'))

    with open(args.dst_path + '/label.txt') as f:
        label_lines = f.readlines()

    print('delete tmp folders')
    #  创建augmentation 文件夹
    for index in tqdm(range(5)):
        raw_path = args.dst_path + '/{0}_of_5folds/train'.format(index + 1)
        train_aug_path = args.dst_path + '/{0}_of_5folds/train_augmentation'.format(index + 1)
        val_aug_path = args.dst_path + '/{0}_of_5folds/val_augmentation'.format(index + 1)
        for line in label_lines:
            shutil.copytree(raw_path + '/' + line.strip('\n'), train_aug_path + '/' + line.strip('\n'))  # 原图文件夹复制到augmentation
            os.makedirs(val_aug_path + '/' + line.strip('\n'))
        shutil.rmtree(args.dst_path + "/{0}_of_5".format(index + 1))