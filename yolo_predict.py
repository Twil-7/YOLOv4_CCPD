import colorsys
import os
import time
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from nets.yolov4_model import yolo_body
from decode import yolo_eval


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


num_anchors = 9
num_classes = 1
image_input = Input(shape=(416, 416, 3))
model = yolo_body(image_input, num_anchors // 3, num_classes)
model.load_weights('best_ep036-loss0.390-val_loss0.381.h5')

anchors_path = 'model_data/yolo_anchors.txt'
anchors = get_anchors(anchors_path)

classes_path = 'model_data/new_classes.txt'
class_names = get_classes(classes_path)    # ['car']

test_path = 'model_data/2021_test.txt'

with open(test_path) as f:
    test_lines = f.readlines()

t1 = time.time()
for i in range(len(test_lines)):

    annotation_line = test_lines[i]
    line = annotation_line.split()

    img1 = Image.open(line[0])
    image_shape = (img1.height, img1.width)    # (520, 660)

    img2 = img1.resize((416, 416))
    img3 = np.array(img2, dtype='float32')
    img4 = img3 / 255.
    img5 = np.expand_dims(img4, 0)

    # img_cv = cv2.cvtColor(img4, cv2.COLOR_RGB2BGR)
    # cv2.namedWindow("Image")
    # cv2.imshow("Image", img_cv)
    # cv2.waitKey(0)

    result1 = model.predict(img5)
    # print(len(result1))        # 3
    # print(result1[0].shape)    # (1, 13, 13, 18)
    # print(result1[1].shape)    # (1, 26, 26, 18)
    # print(result1[2].shape)    # (1, 52, 52, 18)

    boxes_, scores_, classes_ = yolo_eval(result1, anchors, num_classes, image_shape,
                                          score_threshold=0.1, iou_threshold=0.3)

    boxes_1 = np.array(boxes_)        # [[283.2814  110.70488 376.08636 366.71674]]
    scores_1 = np.array(scores_)      # [0.55056524]
    classes_1 = np.array(classes_)    # [0]

    detect_img = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)

    if len(boxes_1) > 0:
        for k in range(len(boxes_1)):

            b1 = int(boxes_1[k, 0])
            a1 = int(boxes_1[k, 1])
            b2 = int(boxes_1[k, 2])
            a2 = int(boxes_1[k, 3])

            index = int(classes_1[k])
            pre_class = str(class_names[index])
            pre_score = round(scores_1[k], 2)    # 保留两位小数
            pre_score = str(pre_score)

            text = pre_class + ': ' + pre_score

            cv2.rectangle(detect_img, (a1, b1), (a2, b2), (0, 0, 255), 2)
            cv2.putText(detect_img, text, (a1, b1), 1, 2, (0, 0, 255))

    # cv2.namedWindow("detect_img")
    # cv2.imshow("detect_img", detect_img)
    # cv2.waitKey(0)

    # cv2.imwrite("demo/" + str(i) + '.jpg', detect_img/1.0)

t2 = time.time()
print((t2 - t1) / len(test_lines))


