#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,Dropout
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model, load_model



# In[ ]:


# TODO 设置正则化系数
global_regulizer = keras.regularizers.l2(0.00001)
dense_regulizer =  keras.regularizers.l2(0.0001)


# ###  1. 定义卷积层

# In[17]:


def conv_block(X, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor 
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    conv_name_base = 'conv' + str(stage) + '_' +'block'+ str(branch) + '_'+ '{0}'+ '_' + 'conv'
    bn_name_base = 'conv' + str(stage) + '_' +'block'+ str(branch) + '_'+ '{0}'+ '_' + 'bn'
    relu_name_base = 'conv' + str(stage) + '_' +'block'+ str(branch) + '_'+ '{0}'+ '_' + 'relu'

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4  
    X = BatchNormalization(axis = 3, name=bn_name_base.format(0))(X)
    X = Activation('relu',name = relu_name_base.format(0))(X)
    X = Conv2D(inter_channel, (1,1), name = conv_name_base.format(1),use_bias=False, kernel_regularizer=global_regulizer)(X)
    ###
    if dropout_rate:
        X = Dropout(dropout_rate)(X)
    # 3x3 Convolution
    X = BatchNormalization(axis = 3, name=bn_name_base.format(1))(X)
    X = Activation('relu',name = relu_name_base.format(1))(X)
    X = Conv2D(nb_filter, (3, 3), name = conv_name_base.format(2), padding='same', use_bias=False, kernel_regularizer=global_regulizer)(X)
    
    if dropout_rate:
        X = Dropout(dropout_rate)(X)

    return X


# ###  2. 定义转换层

# In[18]:


def transition_block(X, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'pool' + str(stage) + '_conv'
    bn_name_base = 'pool' + str(stage) + '_bn'
    relu_name_base = 'pool' + str(stage) + '_relu'
    pool_name_base = 'pool' + str(stage) + '_pool'

    X = BatchNormalization(axis = 3, name=bn_name_base)(X)
    X = Activation('relu', name=relu_name_base)(X)
    X = Conv2D(int(nb_filter * compression), (1,1), name = conv_name_base, use_bias=False, kernel_regularizer=global_regulizer)(X)
    if dropout_rate:
        X = Dropout(dropout_rate)(X)

    X = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(X)

    return X


# ###  3. 定义dense_block

# In[19]:

def se_block(x, stage,radio=4):
    shape = x.get_shape().as_list()
    channel_out = shape[3]
    pool_name = 'SEblock_stage{0}_pool'.format(stage)
    dense_name = 'SEblock_stage{0}_dense_{1}'
    
    x_radio = layers.GlobalAveragePooling2D(name = pool_name)(x)
    x_radio = Flatten()(x_radio)
    
    x_radio = Dense(int(channel_out/radio),kernel_regularizer=global_regulizer,name = dense_name.format(stage,1))(x_radio)
    x_radio = Dense(channel_out, kernel_regularizer=global_regulizer, name = dense_name.format(stage,2))(x_radio)
    x_radio = tf.reshape(x_radio, [-1, 1, 1, channel_out])
    
    x_radio = x_radio * x 
    return x_radio

def dense_block(X, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    concat_feat = X

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_name = 'conv{0}_block{1}_concat'.format(stage, branch)
        concat_feat = keras.layers.concatenate(inputs=[concat_feat, x], axis=3, name=concat_name)
        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter


# In[20]:


def DenseNet121(nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, classes=1000, weights_path=None):
    
    X_input = Input((224,224, 3))
    X = X_input
    compression = 1.0 - reduction
    nb_dense_block = 4
    nb_filter = 64
    nb_layers = [6,12,24,16] # For DenseNet-121
    # stage 1 
    X = ZeroPadding2D((3,3))(X)
    X = Conv2D(nb_filter, (7,7), name='conv1/conv',strides=(2,2),use_bias = False,kernel_regularizer=global_regulizer)(X)    
    X = BatchNormalization(axis = 3, name = 'conv1/bn')(X)
    X = Activation('relu', name = 'conv1/relu')(X)
    
    X = ZeroPadding2D((1,1))(X)
    X = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding = 'valid',name='pool1')(X)
    
    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        X, nb_filter = dense_block(X, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
        # Add transition_block
        X = transition_block(X, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)
    final_stage = stage + 1
    X, nb_filter = dense_block(X, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
    
    X = BatchNormalization(axis = 3, name='bn')(X)
    X = Activation('relu', name='relu')(X)
    X = keras.layers.GlobalAvgPool2D()(X)
    X = Dense(classes, name = 'fc' + str(classes), activation=None,kernel_regularizer=global_regulizer)(X)
    model = keras.Model(inputs = X_input, outputs = X, name = 'DenseNet121')
    
    if weights_path is not None:
        model.load_weights(weights_path)
    
    return model


# In[25]:

def DenseNet_lighter(nb_dense_block=4, growth_rate=12, nb_filter=64, reduction=0.2, dropout_rate=0.2, weight_decay=1e-4, classes=1000, weights_path=None):
    X_input = Input((224,224, 3))
    X = X_input
    compression = 1.0 - reduction
    nb_dense_block = 4
    nb_filter = nb_filter
    nb_layers = [2,4,6,4] # For DenseNet-121
    # stage 1 
    X = ZeroPadding2D((3,3))(X)
    X = Conv2D(nb_filter, (7,7), name='conv1/conv',strides=(2,2),use_bias = False,kernel_regularizer=global_regulizer)(X)    
    X = BatchNormalization(axis = 3, name = 'conv1/bn')(X)
    X = Activation('relu', name = 'conv1/relu')(X)
    
    X = ZeroPadding2D((1,1))(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool1')(X)
    
    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        X, nb_filter = dense_block(X, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
        # Add transition_block
        X = transition_block(X, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)
    final_stage = stage + 1
    X, nb_filter = dense_block(X, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
    
    X = BatchNormalization(axis = 3, name='bn')(X)
    X = Activation('relu', name='relu')(X)
    X = keras.layers.GlobalAvgPool2D()(X)
    X = Dense(classes, name = 'fc' + str(classes), activation='softmax',kernel_regularizer=dense_regulizer)(X)
    model = keras.Model(inputs = X_input, output =X, name= 'DenseNet_lighter')
    
    if weights_path is not None:
        model.load_weights(weights_path)
    
    return model

def DenseNet_SE_lighter(nb_dense_block=4, growth_rate=12, nb_filter=64, reduction=0.2, dropout_rate=0.2, weight_decay=1e-4, classes=1000, weights_path=None):
    X_input = Input((224,224, 3))
    X = X_input
    compression = 1.0 - reduction
    nb_dense_block = 4
    nb_filter = nb_filter
    nb_layers = [2,4,6,4] # For DenseNet-121
    # stage 1 
    X = ZeroPadding2D((3,3))(X)
    X = Conv2D(nb_filter, (7,7), name='conv1/conv',strides=(2,2),use_bias = False,kernel_regularizer=global_regulizer)(X)    
    X = BatchNormalization(axis = 3, name = 'conv1/bn')(X)
    X = Activation('relu', name = 'conv1/relu')(X)
    
    X = ZeroPadding2D((1,1))(X)
    X = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding = 'valid',name='pool1')(X)
    
    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        X, nb_filter = dense_block(X, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
        # Add transition_block
        X = se_block(X, stage)
        X = transition_block(X, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)
    final_stage = stage + 1
    X, nb_filter = dense_block(X, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
    
    X = BatchNormalization(axis = 3, name='bn')(X)
    X = Activation('relu', name='relu')(X)
    # X = keras.layers.GlobalAvgPool2D()(X)
    X = AveragePooling2D(pool_size=(2, 2))(X)
    X = Flatten()(X)
    #
    X = Dense(classes, name = 'fc' + str(classes), activation='sigmoid',kernel_regularizer=dense_regulizer)(X)
    model = keras.Model(inputs = X_input, outputs=X, name='DenseNet_lighter')
    
    if weights_path is not None:
        model.load_weights(weights_path)
    
    return model


# ###  4. 定义预处理函数

# In[26]:


if __name__ == '__main__':
    model = DenseNet_lighter(reduction=0.5, classes = 2)
    model.summary()


# In[ ]:




