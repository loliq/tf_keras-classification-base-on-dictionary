#!/usr/bin/env python
# coding: utf-8

# mobileNet参数说明：
# - depth_multiplier: 在Depthwise Conv用的控制网络大小的参数，一般为1
# - alpha：按比例缩减网络每一层的参数
# - 与官方给的不同的是我没加Zero_padding，卷积的padding统一使用的是"same_padding"

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, BatchNormalization,Activation, Dropout,ZeroPadding2D, DepthwiseConv2D,add,Dense,GlobalAveragePooling2D
from tensorflow.keras.initializers import glorot_uniform


# In[2]:


def _conv_block(x_input, filters, kernel, stride, kernel_regulizer, dropout_rate, base_name):
    conv_name_base = base_name
    bn_name_base =  base_name + '_' + 'BN'
    relu_name_base = base_name + "_" + 'relu'
    x = Conv2D(filters, kernel_size=kernel,strides=stride, kernel_regularizer=kernel_regulizer, 
               name=conv_name_base,padding='same', use_bias=False)(x_input)
    
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = BatchNormalization(name=bn_name_base)(x)
    x = Activation(tf.nn.relu6, name=relu_name_base)(x)
    return x


# In[3]:


def _bottleneck(x_input, filters, kernel, t, s, kernel_regulizer, 
                dropout_rate,base_name,depth_multiplier=1, r=False):
    """
    """
    
    t_channels = x_input.shape[3] * 6
    x = _conv_block(x_input, t_channels, kernel=(1,1), stride=(1,1), kernel_regulizer=kernel_regulizer,
                    dropout_rate=dropout_rate, base_name=base_name+'_expand')
#     x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    # TODO depth_multiper可以作为改的
    x = DepthwiseConv2D(kernel_size=kernel, strides=(s, s), depth_multiplier=1,
                        padding='same', name=base_name+'_depthwise',use_bias=False)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = BatchNormalization(name=base_name+'_depthwise' + '_BN')(x)
    x = Activation(tf.nn.relu6, name=base_name+'_depthwise' + '_relu')(x)
    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name=base_name+'_project', use_bias=False)(x)
    
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = BatchNormalization(name=base_name+'_project' + '_BN')(x)
    if r:
        x = add([x, x_input], name=base_name + '_add')
    return x


# BottleNeck的结构

# ![image.png](https://img-blog.csdnimg.cn/20181220125051819.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTE5NzQ2Mzk=,size_16,color_FFFFFF,t_70)

# In[4]:


def _inverted_residual_block(inputs, filters, kernel, t, strides, n, 
                             kernel_regulizer, dropout_rate, depth_multiplier, init_block_id=1):
    
    """Inverted Residual Block
    
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, strides, 
                    kernel_regulizer=kernel_regulizer, dropout_rate=dropout_rate, 
                    depth_multiplier=depth_multiplier, base_name='block_' + str(init_block_id))
    

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, 
                    kernel_regulizer=kernel_regulizer, dropout_rate=dropout_rate, 
                    depth_multiplier=depth_multiplier, base_name='block_' + str(init_block_id + i),r=True)

    return x


# In[5]:


def build_mobileNetV2(input_shape=[224, 224, 3],
                      l2_regularizer_weight=0.0001,
                      dropout_rate=None,
                      depth_multiplier=1,
                      class_num=2, 
                      alpha= 1):
    global_regulizer = keras.regularizers.l2(l2_regularizer_weight)
    X_input = layers.Input(shape=input_shape, name="input")
    # X = ZeroPadding2D(((1, 0), (1, 0)))(X_input)
    X = _conv_block(X_input, filters=int(32 * alpha), kernel=(3,3), base_name='conv1', stride=2, 
                    dropout_rate=dropout_rate, kernel_regulizer=global_regulizer)
    X = DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), depth_multiplier=1, padding='same', 
                        name='expanded_conv_depthwise',use_bias=False)(X)
    X = BatchNormalization(name='expanded_conv_depthwise_BN')(X)
    X = Activation(tf.nn.relu6, name='expanded_conv_depthwise_relu')(X)
    X = Conv2D(filters= int(16 * alpha), kernel_size=(1,1), strides=(1,1), padding='same', name='expanded_conv_project', use_bias=False)(X)
    X = BatchNormalization(name='expanded_conv_project_BN')(X)
    # invert residul block
    X = _inverted_residual_block(X, filters=int(24 * alpha), kernel=(3,3), t=6, strides=2, n=2,
                                 dropout_rate=dropout_rate, kernel_regulizer=global_regulizer,
                                 depth_multiplier=depth_multiplier,init_block_id=1)
    X = _inverted_residual_block(X, filters=int(32 * alpha), kernel=(3,3), t=6, strides=2, n=3,
                                 dropout_rate=dropout_rate, kernel_regulizer=global_regulizer,
                                 depth_multiplier=depth_multiplier,init_block_id=3)
    X = _inverted_residual_block(X, filters=int(64 * alpha), kernel=(3,3), t=6, strides=2, n=4,
                                 dropout_rate=dropout_rate, kernel_regulizer=global_regulizer,
                                 depth_multiplier=depth_multiplier,init_block_id=6)
    X = _inverted_residual_block(X, filters=int(96 * alpha), kernel=(3,3), t=6, strides=1, n=3,
                                 dropout_rate=dropout_rate, kernel_regulizer=global_regulizer,
                                 depth_multiplier=1,init_block_id=10)
    X = _inverted_residual_block(X, filters=int(160 * alpha), kernel=(3,3), t=6, strides=2, n=3,
                                 dropout_rate=dropout_rate, kernel_regulizer=global_regulizer,
                                 depth_multiplier=depth_multiplier,init_block_id=13)
    X = _inverted_residual_block(X, filters=int(320 * alpha), kernel=(3,3), t=6, strides=1, n=1,
                                 dropout_rate=dropout_rate, kernel_regulizer=global_regulizer,
                                 depth_multiplier=depth_multiplier,init_block_id=16)
    # 卷积
    X = _conv_block(X, filters= 1280, kernel=(1,1), base_name='conv2', stride=1, 
                    dropout_rate=dropout_rate, kernel_regulizer=global_regulizer)
    # 分类层
    X = GlobalAveragePooling2D()(X)
    X = Dense(class_num, name="output")(X)
    model = keras.Model(inputs=X_input, outputs=X, name='mobileNet')
    return model


# In[6]:


# model = build_mobileNetV2(alpha=0.5)
# model.summary()


# In[14]:


class mobilenetV2(object):
    def __init__(self, kwargs):
        self.__dict__.update(kwargs)

    def constuct_model(self):

        self.model = build_mobileNetV2(input_shape=self.input_shape,
                                       class_num=self.classes,
                                       dropout_rate=self.dropout_rate,
                                       l2_regularizer_weight=self.l2_regularizer_weight,
                                       alpha=self.alpha,
                                       depth_multiplier=self.depth_multiplier)
        return self.model


# In[15]:


if __name__ == '__main__':
    model_config=dict(
        type='mobilenetV2',  #模型类中的较小的构造模型的函数
        input_shape=[224, 224, 1],
        classes=2,
        dropout_rate=0.2,
        l2_regularizer_weight=0.0001,
        alpha=1,
        depth_multiplier=1)
    base_model = mobilenetV2(model_config).build_model()
    base_model.summary()






