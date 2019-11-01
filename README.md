# tf_keras-classification-base-on-dictionary


使用tf.kera 搭配eager_execution来分类，
- 主要作用是不用再额外做标签文件，可以直接将文件夹看成一个类(有些时候调整标签，增加数据啥的还要重新生成一堆标签文件可太烦了)。然后定义标签映射文件进行分类。
- 在cls_model 类中还定义了从config和权重文件如何转pb的函数

## 
- python3.6
- tf.1.14

## 训练
[train.py](https://github.com/loliq/tf_keras-classification-base-on-dictionary/blob/master/train.py). 
目前暂时还是用命令行参数替代, 后期有空会写配置文件格式
参数意义
> 1. --logdir: 放置训练过程中产生的模型，默认会存网络模型的结构文件(.json文件)
> 2. --train_folder_path: 存放图片文件的路径，路径下有以类别为命名的文件夹。
> 3. -- val_folder_path: 存放验证图片文件的路径，路径下有以类别为命名的文件夹。
> 4. -- label_path: 以字典形式存放的标签类别映射。例：
```
{"边缘OK":0,"视窗OK":0,"凹凸压痕":1,"崩边":2, "边缘不良":2, "扫边OK":3, "尘点":4,"边波纹": 5, "边缘可过不良":6,"扫边OK":7}
```
> 5. -- batch_size
> 6. --class_num
> 7. --load_pretrained
> 8. --model_config_path: 存放模型的配置文件(.json文件)
> 9. --model_path: 存放模型的权重文件

## losses_and__metrics.py
存放一些自定义的loss函数和评估函数
- 目前有放一些focal_loss的函数
## utils.py
存放组合dataset数据通道的函数及实现(预处理也在这里做，因此预处理函数需要在这里调用)

--------------------------待更新, 定义preprocess函数，然后将函数以参数形式传入------------------

## cls_model.py
一些方法定义及实现
## models
存放一些经典模型
目前有ResNet,DenseNer



