import os
from PIL import Image
import numpy as np

# k_means虽然会对数据集中的框进行聚类，但是很多数据集由于框的大小相近，聚类出来的9个框相差不大，这样的框反而不利于模型的训练。
# 因为不同的特征层适合不同大小的先验框，越浅的特征层适合越大的先验框。
# 原始网络的先验框已经按大中小比例分配好了，不进行聚类也会有非常好的效果。


def cas_iou(box, cluster):

    x = np.minimum(cluster[:, 0], box[0])
    y = np.minimum(cluster[:, 1], box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:, 0] * cluster[:, 1]
    iou = intersection / (area1 + area2 - intersection)

    return iou


def avg_iou(box, cluster):

    return np.mean([np.max(cas_iou(box[i], cluster)) for i in range(box.shape[0])])


def k_means(box, k):

    # 取出一共有多少框
    row = box.shape[0]
    
    # 每个框各个点的位置
    distance = np.empty((row, k))
    
    # 最后的聚类位置
    last_clu = np.zeros((row,))

    np.random.seed(10)

    # 随机选5个当聚类中心
    cluster = box[np.random.choice(row, k, replace=False)]

    while True:

        # 计算每一行距离9个点的iou情况
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i], cluster)
        
        # 取出最小点
        near = np.argmin(distance, axis=1)

        if (last_clu == near).all():
            break
        
        # 求每一个类的中位点
        for j in range(k):
            cluster[j] = np.median(box[near == j], axis=0)

        last_clu = near

    return cluster


if __name__ == '__main__':

    SIZE = 416
    anchors_num = 9

    data = []
    filename = os.listdir('crop_ccpd')
    filename.sort()
    for name in filename:

        img_path = 'crop_ccpd/' + name
        img = Image.open(img_path)
        width = img.width
        height = img.height

        if height <= 0 or width <= 0:
            continue

        obj1 = name.split('.')
        obj2 = obj1[0]
        obj3 = obj2.split('_')

        x1 = np.float64(int(obj3[1]) / width)
        y1 = np.float64(int(obj3[2]) / height)
        x2 = np.float64(int(obj3[3]) / width)
        y2 = np.float64(int(obj3[4]) / height)

        data.append([x2 - x1, y2 - y1])

    data = np.array(data)

    # 使用k聚类算法
    out = k_means(data, anchors_num)
    out = out[np.argsort(out[:, 0])]
    print('acc:{:.2f}%'.format(avg_iou(data, out) * 100))
    print('anchors :', out*SIZE)

    anchors = out*SIZE
    f = open("./model_data/yolo_anchors1.txt", 'w')
    r = np.shape(anchors)[0]
    for i in range(r):
        if i == 0:
            x_y = "%d,%d" % (anchors[i][0], anchors[i][1])
        else:
            x_y = ", %d,%d" % (anchors[i][0], anchors[i][1])
        f.write(x_y)
    f.close()
