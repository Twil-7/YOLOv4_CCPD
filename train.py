from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

from nets.loss import yolo_loss
from nets.yolov4_model import yolo_body
from data_generate import data_generator


def get_classes(path):

    with open(path) as f:
        cls_names = f.readlines()
    cls_names = [c.strip() for c in cls_names]
    return cls_names


def get_anchors(path):

    with open(path) as f:
        anchor = f.readline()
    anchor = [float(x) for x in anchor.split(',')]

    return np.array(anchor).reshape(-1, 2)


if __name__ == "__main__":

    train_path = 'model_data/2021_train.txt'
    val_path = 'model_data/2021_val.txt'
    test_path = 'model_data/2021_test.txt'

    log_dir = 'Logs/'
    classes_path = 'model_data/new_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'

    weights_path = 'model_data/yolo4_weight.h5'
    label_smoothing = 0
    input_shape = (416, 416)

    class_names = get_classes(classes_path)    # ['car']
    anchors = get_anchors(anchors_path)
    # [[114.  53.]
    #  [139.  64.]
    #  [147.  79.]
    #  [164.  71.]
    #  [170.  95.]
    #  [189.  82.]
    #  [197. 108.]
    #  [221.  94.]
    #  [243. 119.]]

    num_classes = len(class_names)    # 1
    num_anchors = len(anchors)        # 9

    image_input = Input(shape=(416, 416, 3))
    h, w = input_shape

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    model_body.summary()
    model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)

    #   搭建损失函数层，将网络的输出结果传入loss函数，把整个模型的输出作为loss
    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                           num_anchors // 3, num_classes + 5)) for l in range(3)]
    loss_input = [*model_body.output, *y_true]

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes,
                                   'ignore_thresh': 0.5, 'label_smoothing': label_smoothing})(loss_input)

    model = Model([model_body.input, *y_true], model_loss)

    checkpoint = ModelCheckpoint(log_dir + "/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                                 save_weights_only=True, save_best_only=False, period=1)

    with open(train_path) as f:
        train_lines = f.readlines()

    with open(val_path) as f:
        val_lines = f.readlines()

    num_train = len(train_lines)    # 2900
    num_val = len(val_lines)        # 100

    freeze_layers = 249
    for i in range(freeze_layers):
        model_body.layers[i].trainable = False

    Init_epoch = 0
    Freeze_epoch = 250
    batch_size = 5
    learning_rate_base = 1e-3

    epoch_size_train = num_train // batch_size    # 580
    epoch_size_val = num_val // batch_size        # 20

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    model.compile(optimizer=Adam(learning_rate_base), loss={'yolo_loss': lambda y_true, y_pre: y_pre})

    # 训练过程中random数据增强对模型效果提升非常重要，可大幅降低val loss

    model.fit(data_generator(train_lines, batch_size, input_shape, anchors, num_classes, random=True),
              steps_per_epoch=epoch_size_train,
              validation_data=data_generator(val_lines, batch_size, input_shape, anchors, num_classes, random=False),
              validation_steps=epoch_size_val,
              epochs=Freeze_epoch,
              initial_epoch=Init_epoch,
              callbacks=[checkpoint, reduce_lr])
