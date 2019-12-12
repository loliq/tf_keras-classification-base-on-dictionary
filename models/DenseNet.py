#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/12 16:23
# @Author  : LLL
# @Site    : 
# @File    : DenseNet1.py
# @Software: PyCharm

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

tf.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model, load_model


class DenseNet(object):
    def __init__(self, kwargs):
        self.__dict__.update(kwargs)

    def constuct_model(self):
        if self.type == "DenseNet_lighter":
            model_func = self.DenseNet_lighter
        if self.type == "DenseNet121":
            model_func = self.DenseNet121

        model = model_func(nb_dense_block=self.nb_dense_block,
                           nb_layers=self.nb_layers,
                           input_shape=self.input_shape,
                           classes=self.classes,
                           growth_rate=self.growth_rate,
                           nb_filter=self.nb_filter,
                           reduction=self.reduction,
                           dropout_rate=self.dropout_rate,
                           l2_regularizer_weight=self.l2_regularizer_weight,
                           weight_decay=self.weight_decay)
        return model

    def DenseNet121(self,
                    nb_dense_block=4,
                    nb_layers=[6, 12, 24, 16],
                    classes=1000,
                    nb_filter=64,
                    Input_shape=[224, 224, 3],
                    growth_rate=32,
                    reduction=0.0,
                    dropout_rate=0.0,
                    l2_regularizer_weight=0.0001,
                    weight_decay=1e-4):

        X_input = Input(Input_shape)
        global_regulizer = keras.regularizers.l2(l2_regularizer_weight)
        X = X_input
        compression = 1.0 - reduction
        # stage 1
        X = ZeroPadding2D((3, 3))(X)
        X = Conv2D(nb_filter, (7, 7), name='conv1/conv', strides=(2, 2), use_bias=False,
                   kernel_regularizer=global_regulizer)(X)
        X = BatchNormalization(axis=3, name='conv1/bn')(X)
        X = Activation('relu', name='conv1/relu')(X)

        X = ZeroPadding2D((1, 1))(X)
        X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool1')(X)

        # Add dense blocks
        for block_idx in range(nb_dense_block - 1):
            stage = block_idx + 2
            X, nb_filter = self.dense_block(X, stage, nb_layers[block_idx], nb_filter, growth_rate,
                                            dropout_rate=dropout_rate, weight_decay=weight_decay)
            # Add transition_block
            X = self.transition_block(X, stage, nb_filter, compression=compression, dropout_rate=dropout_rate,
                                      weight_decay=weight_decay)
            nb_filter = int(nb_filter * compression)
        final_stage = stage + 1
        X, nb_filter = self.dense_block(X, final_stage, nb_layers[-1], nb_filter, growth_rate,
                                        dropout_rate=dropout_rate, weight_decay=weight_decay)

        X = BatchNormalization(axis=3, name='bn')(X)
        X = Activation('relu', name='relu')(X)
        X = keras.layers.GlobalAvgPool2D()(X)
        X = Dense(classes, name='fc' + str(classes), activation=None, kernel_regularizer=global_regulizer)(X)
        model = keras.Model(inputs=X_input, outputs=X, name='DenseNet121')
        return model

    def DenseNet_lighter(self, nb_dense_block=4, nb_layers=[2, 4, 6, 4],
                         input_shape=[224, 224, 3],
                         classes=1000,
                         growth_rate=12, nb_filter=64,
                         reduction=0.2, dropout_rate=0.2,
                         l2_regularizer_weight=0.0001,
                         weight_decay=1e-4,
                         ):
        global_regulizer = keras.regularizers.l2(l=l2_regularizer_weight)
        X_input = Input(input_shape)
        X = X_input
        compression = 1.0 - reduction
        nb_dense_block = nb_dense_block
        nb_filter = nb_filter
        nb_layers = nb_layers
        # stage 1
        X = ZeroPadding2D((3, 3))(X)
        X = Conv2D(nb_filter, (7, 7), name='conv1/conv', strides=(2, 2), use_bias=False,
                   kernel_regularizer=global_regulizer)(X)
        X = BatchNormalization(axis=3, name='conv1/bn')(X)
        X = Activation('relu', name='conv1/relu')(X)

        X = ZeroPadding2D((1, 1))(X)
        X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool1')(X)

        # Add dense blocks
        for block_idx in range(nb_dense_block - 1):
            stage = block_idx + 2
            X, nb_filter = self.dense_block(X, stage, nb_layers[block_idx],
                                            nb_filter, growth_rate,
                                            dropout_rate=dropout_rate,
                                            global_regulizer=global_regulizer,
                                            weight_decay=weight_decay)
            # Add transition_block
            X = self.transition_block(X, stage, nb_filter,
                                      compression=compression,
                                      dropout_rate=dropout_rate,
                                      global_regulizer=global_regulizer,
                                      weight_decay=weight_decay)
            nb_filter = int(nb_filter * compression)
        final_stage = stage + 1
        X, nb_filter = self.dense_block(X, final_stage, nb_layers[-1],
                                        nb_filter, growth_rate,
                                        dropout_rate=dropout_rate,
                                        global_regulizer=global_regulizer,
                                        weight_decay=weight_decay)

        X = BatchNormalization(axis=3, name='bn')(X)
        X = Activation('relu', name='relu')(X)
        X = keras.layers.GlobalAvgPool2D()(X)
        X = Dense(classes, name='fc' + str(classes), activation='softmax', kernel_regularizer=global_regulizer)(X)
        model = keras.Model(inputs=X_input, outputs=X, name='DenseNet_lighter')

        return model

    def conv_block(self, X, stage, branch, nb_filter, global_regulizer, dropout_rate=None, weight_decay=1e-4):
        '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
            # Arguments
                x: input tensor
                stage: index for dense block
                branch: layer index within each dense block
                nb_filter: number of filters
                dropout_rate: dropout rate
                weight_decay: weight decay factor
        '''
        conv_name_base = 'conv' + str(stage) + '_' + 'block' + str(branch) + '_' + '{0}' + '_' + 'conv'
        bn_name_base = 'conv' + str(stage) + '_' + 'block' + str(branch) + '_' + '{0}' + '_' + 'bn'
        relu_name_base = 'conv' + str(stage) + '_' + 'block' + str(branch) + '_' + '{0}' + '_' + 'relu'

        # 1x1 Convolution (Bottleneck layer)
        inter_channel = nb_filter * 4
        X = BatchNormalization(axis=3, name=bn_name_base.format(0))(X)
        X = Activation('relu', name=relu_name_base.format(0))(X)
        X = Conv2D(inter_channel, (1, 1), name=conv_name_base.format(1), use_bias=False,
                   kernel_regularizer=global_regulizer)(X)
        ###
        if dropout_rate:
            X = Dropout(dropout_rate)(X)
        # 3x3 Convolution
        X = BatchNormalization(axis=3, name=bn_name_base.format(1))(X)
        X = Activation('relu', name=relu_name_base.format(1))(X)
        X = Conv2D(nb_filter, (3, 3), name=conv_name_base.format(2), padding='same', use_bias=False,
                   kernel_regularizer=global_regulizer)(X)

        if dropout_rate:
            X = Dropout(dropout_rate)(X)
        return X

    def transition_block(self, X, stage, nb_filter, global_regulizer,
                         compression=1.0, dropout_rate=None, weight_decay=1E-4):
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

        X = BatchNormalization(axis=3, name=bn_name_base)(X)
        X = Activation('relu', name=relu_name_base)(X)
        X = Conv2D(int(nb_filter * compression), (1, 1), name=conv_name_base, use_bias=False,
                   kernel_regularizer=global_regulizer)(X)
        if dropout_rate:
            X = Dropout(dropout_rate)(X)

        X = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(X)

        return X

    def dense_block(self, X, stage, nb_layers,
                    nb_filter, growth_rate,
                    global_regulizer,
                    dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
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
            branch = i + 1
            x = self.conv_block(concat_feat, stage, branch, growth_rate, global_regulizer, dropout_rate, weight_decay)
            concat_name = 'conv{0}_block{1}_concat'.format(stage, branch)
            concat_feat = keras.layers.concatenate(inputs=[concat_feat, x],
                                                   axis=3, name=concat_name)
            if grow_nb_filters:
                nb_filter += growth_rate

        return concat_feat, nb_filter