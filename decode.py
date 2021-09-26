import os
import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from nets.loss import yolo_head


#   获取每个grid的box和它的得分
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):

    # feats.shape : (1, 13, 13, 18)
    # anchors :
    # [[197. 108.]
    #  [221.  94.]
    #  [243. 119.]]

    box_xy, box_wh, box_confidence, box_class_prob = yolo_head(feats, anchors, num_classes, input_shape)
    # box_xy.shape : (1, 13, 13, 3, 2)
    # box_wh.shape : (1, 13, 13, 3, 2)
    # box_confidence.shape : (1, 13, 13, 3, 1)
    # box_class_prob.shape : (1, 13, 13, 3, 1)

    # 交换x和y、w和h的列位置
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    box_min = box_yx - (box_hw / 2.)
    box_max = box_yx + (box_hw / 2.)

    image_shape = K.cast(image_shape, K.dtype(box_yx))

    # boxes : y_min, x_min, y_min, x_min, (1, 13, 13, 3, 4)

    boxes = K.concatenate([box_min[..., 0:1] * image_shape[0],
                           box_min[..., 1:2] * image_shape[1],
                           box_max[..., 0:1] * image_shape[0],
                           box_max[..., 1:2] * image_shape[1]])

    #   获得最终得分和框的位置
    boxes = K.reshape(boxes, [-1, 4])                         # (507, 4)
    box_scores = box_confidence * box_class_prob              # (1, 13, 13, 3, 1)
    box_scores = K.reshape(box_scores, [-1, num_classes])     # (507, 1)

    return boxes, box_scores


def yolo_eval(yolo_outputs, anchors, num_classes, image_shape, max_boxes=20, score_threshold=0.1, iou_threshold=0.3):

    # yolo_outputs[0].shape : (1, 13, 13, 18)
    # yolo_outputs[1].shape : (1, 26, 26, 18)
    # yolo_outputs[2].shape : (1, 52, 52, 18)

    num_layers = len(yolo_outputs)    # 3

    #   13x13的特征层对应的anchor是[142, 110], [192, 243], [459, 401]
    #   26x26的特征层对应的anchor是[36, 75], [76, 55], [72, 146]
    #   52x52的特征层对应的anchor是[12, 16], [19, 36], [40, 28]

    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    #   这里获得的是输入图片的大小，一般是416x416
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []

    #   对每个特征层进行处理
    for l in range(num_layers):

        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]],
                                                    num_classes, input_shape, image_shape)
        # _boxes.shape : (507, 4)
        # _box_scores.shape : (507, 4)

        boxes.append(_boxes)
        box_scores.append(_box_scores)

    # 将每个特征层的结果进行堆叠
    boxes = K.concatenate(boxes, axis=0)              # (10647, 4)
    box_scores = K.concatenate(box_scores, axis=0)    # (10647, 4)

    # 忽略掉所有可信度小于阈值 score_threshold 的边框

    mask = box_scores >= score_threshold    # (10647, 1)
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')

    boxes_ = []
    scores_ = []
    classes_ = []

    # 对每个类别的所有边框，都进行一遍非极大抑制
    for c in range(num_classes):

        # 取出所有box_scores >= score_threshold的框，和score
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        # class_boxes :
        # tf.Tensor(
        # [[283.2814   110.70488  376.08636  366.71674 ]
        #  [280.66226  115.42855  378.71503  359.82193 ]
        #  [281.06467  115.617455 380.11658  358.29422 ]], shape=(3, 4))

        # class_box_scores :
        # tf.Tensor([0.55056524 0.12602141 0.29380256], shape=(3,))

        # 非极大抑制, 保留一定区域内得分最大的框
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores,
                                                 max_boxes_tensor, iou_threshold=iou_threshold)
        # nms_index : tf.Tensor([0], shape=(1,))

        # 获取非极大抑制后的结果: 框的位置，得分，种类

        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c

        # class_boxes : tf.Tensor([[283.2814  110.70488 376.08636 366.71674]], shape=(1, 4))
        # class_box_scores : tf.Tensor([0.55056524], shape=(1,))
        # classes : tf.Tensor([0], shape=(1,))

        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)

    boxes_ = K.concatenate(boxes_, axis=0)        # (1, 4)
    scores_ = K.concatenate(scores_, axis=0)      # (1,)
    classes_ = K.concatenate(classes_, axis=0)    # (1,)

    return boxes_, scores_, classes_
