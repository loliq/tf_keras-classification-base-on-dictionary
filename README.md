# tf_keras-classification-base-on-dictionary

-------------20191211更新--------------------
1. 采用 configs/config.py 作为模型和训练的参数控制
2. 增加pb文件转换的代码

-------------20191117更新--------------------
1. 因为使用文件夹会出现keras运行val_dataset的时候值不对很奇怪而且直接从盘里面读取图片很慢的原因，增加了使用tf_record的方法读取文件
2. 增加了记录训练信息的函数
--------------------------------------------

使用tf.kera 搭配eager_execution来分类，
- 主要作用是不用再额外做标签文件，可以直接将文件夹看成一个类(有些时候调整标签，增加数据啥的还要重新生成一堆标签文件可太烦了)。然后定义标签映射文件进行分类。
- 在cls_model 类中还定义了从config和权重文件如何转pb的函数

## 环境
- python3.6
- tf.1.14

## 训练
[train.py](https://github.com/loliq/tf_keras-classification-base-on-dictionary/blob/master/train.py). 
- 只有一个参数就是配置文件的选择
- 需要注意的是配置统一放在'config'文件夹，选路径的时候填写'config/config.py'即可
> 9. --model_path: 存放模型的权重文件

## losses_and__metrics.py
存放一些自定义的loss函数和评估函数
- 目前有放一些focal_loss的函数
- 为了代码编写便利，放了一些keras内置的loss函数的wrapper
## ultis
- [cls_model.py](https://github.com/loliq/tf_keras-classification-base-on-dictionary/blob/master/ultis/classification_model.py): 模型的实现类
- [dataset.py](https://github.com/loliq/tf_keras-classification-base-on-dictionary/blob/master/ultis/dataset.py): 存放组合dataset数据通道的函数及实现(预处理也在这里做，因此预处理函数需要在这里调用)
- [create_tf_record.py](https://github.com/loliq/tf_keras-classification-base-on-dictionary/blob/master/ultis/create_tf_record.py): 给定文件夹和标签映射，生成tf_record文件
- [k_folds_classification.py](https://github.com/loliq/tf_keras-classification-base-on-dictionary/blob/master/ultis/k_folds_classification.py)： 用于给K-folds交叉验证分配图片集
- [convert_to_pb.py](https://github.com/loliq/tf_keras-classification-base-on-dictionary/blob/master/ultis/convert_to_pb.py]): 用于将存好的model_config(模型结构的.json文件) 和模型权重结合 转成'.pb'文件，同时输出节点信息供调用
--------------------------待更新, 定义preprocess函数，然后将函数以参数形式传入------------------
## models
存放一些经典模型
目前有：
> 1.ResNet: 里面有RestNet50, ResNet18, ResNet34的实现
> 2. DenseNet: 里面有DenseNet121， 精简版本自定义的DenseNet结构的实现

# 使用方法
## 1. 将图片分为训练集和测试集(k-folds-validation)
代码为[k_folds_classification.py]目的是将图片分成k-份(k默认为5，也就是80%训练集，20%的测试集)
参数: 
> - `--data_path`: 原图文件夹，其中每一个文件夹放入不同类别的图片
> - `--dst_path`: 目标文件夹，放随机分配k-folds的图片
> - `--num_folds` : 控制分成几份

## 2. tf_record文件制作
参数说明
> 1. `--origin_dir`: 已经分成k-fold的文件夹，里面有`train`和`val`的子文件夹
> 2. `--out_dir`: 用于存放输出的tf_record文件
> 3. `label_map_path`: 标签映射，用于某些情况下可能多个文件夹的标签需要一样的情况，是一个字典形式的文件。如下
{"05边缘OK":0, "11视窗OK":0, "01NG":1, "02凹凸压痕":1, "03崩边":2, "12扫边NG":2, "06边缘不良":2, "04边波纹":2, "10扫边OK":3, "07边缘可过不良":3, "08尘点":4}
> 4. resize_shape: 为了读取更快，可以考虑在做tf_record文件的时候就做好文件缩放，这个代码用的是等比例缩放，长宽比不同则用黑图填充

## 3. 配置文件的编写
- 见例子编写

## 4. 训练
python train.py ----config_path=config/config.py


