from utils import gen_data_txt,write_label_text,trans_dir,gen_data_txt_with_labels,segment_k_fold, compose_to_dataset
import os
import shutil
import argparse
# Create ArgumentParser() object
parser = argparse.ArgumentParser()
# Add argument
parser.add_argument('--data_path', required=True, help='path to original dataset')
parser.add_argument('--dst_path', required=True, help='path to segment dataset')
parser.add_argument('--num_folds', type=int, help='number of cross_validation folds', default=5)
args = parser.parse_args()

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
    for index in range(args.num_folds):
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

    #  删除临时文件夹
    #  创建augmentation 文件夹
    for index in range(5):
        raw_path = args.dst_path + '/{0}_of_5folds/train'.format(index + 1)
        train_aug_path = args.dst_path + '/{0}_of_5folds/train_augmentation'.format(index + 1)
        val_aug_path = args.dst_path + '/{0}_of_5folds/val_augmentation'.format(index + 1)
        for line in label_lines:
            shutil.copytree(raw_path + '/' + line.strip('\n'), train_aug_path + '/' + line.strip('\n'))  # 原图文件夹复制到augmentation
            os.makedirs(val_aug_path + '/' + line.strip('\n'))
        shutil.rmtree(args.dst_path + "/{0}_of_5".format(index + 1))