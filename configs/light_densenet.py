model = dict(
    type="DenseNet",  # 大的模型类
    pretrained=None,
    work_dir="logs",  # log文件和模型存储路径
    # 预训练的模型结构文件'config.json'和权重'model.h5'
    pretrained_config=dict(
        model_confg_path="",
        model_weight_path="",
    ),
    # 模型参数设置
    # 不同的模型的参数不一样
    model_config=dict(
        type='DenseNet_lighter',  #模型类中的较小的构造模型的函数
        input_shape=[224, 224, 1],
        class_num=5,
        nb_filter=32,
        growth_rate=12,
        dropout_rate=0.2,
        l2_regularizer_weight=0.0001,
        reduce_rate=0.8
    ),
    # 训练参数设置
    train_config=dict(
        batch_size=16,
        total_epoches=600,
        base_lr=0.001,
        optimizer=dict(
            type="RMSprop",
            metric="categorical_accuracy",
            rho=0.9,
            momentum=0.0,
            epsilon=1e-7,
            centered=False
        ),
        loss=dict(
            loss_name="multi_category_focal_loss2",  #
            from_logits=False,
            label_smooth=0.0,
            alpha=0.2,
            gamma=2,
            class_num_distribution=[],
        ),
        lr_config=dict(
            policy='epoch',                        # 优化策略
            warmdown='linear',                      # 初始的学习率增加的策略，linear为线性增加
            warmdown_iters=200,                     # 在初始的200次迭代中学习率逐渐减小
            warmdown_period=4,
            warmdown_ratio=0.95,                 # 起始的学习率
        ),                   # 在第8和11个epoch时降低学习率
        callbacks=dict(
                check_point=dict(
                monitor="val_categorical_accuracy",
                saved_model_name="/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-val_acc{val_categorical_accuracy:.3f}.h5"
                ),
        ),
        data_augmentation_ops=dict(
            random_horizontal_flip=dict()
        )
    ),
    # TODO 设置训练/验证数据集
    data=dict(
        train=dict(
            image_dir=r"F:\01-datasets\04-VTC\01-伯恩\Tensroflow\record_file\train",
            label_path=r"F:\01-datasets\04-VTC\01-伯恩\Tensroflow\label_map.txt",
        ),
        val=dict(
            image_dir=r"F:\01-datasets\04-VTC\01-伯恩\Tensroflow\record_file\val",
            label_path=r"F:\01-datasets\04-VTC\01-伯恩\Tensroflow\label_map.txt"
        )

    )
)
