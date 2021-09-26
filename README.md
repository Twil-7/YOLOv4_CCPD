# YOLOv4_CCPD

# 环境配置：

python == 3.6

tensorflow == 2.3.0

opencv == 3.4.2


# 文件说明：

（1）crop_ccpd文件夹：存放CCPD车牌号数据集，含有3116张图片。

下载路径：https://blog.csdn.net/Twilight737?spm=1018.2226.3001.5343&type=download

（2）Logs文件夹：存放训练过程中的权重文件。

（3）model_data：存放划分的数据集path、原始YOLOv4权重、anchors系数、类别信息。

（4）nets文件夹：存放YOLOv4模型.py文件。

（5）utils文件夹：存放YOLOv4辅助功能.py文件。

（5）权重文件：

yolo4_weight.h5：初始载入的YOLOv4权重。

best_ep036-loss0.390-val_loss0.381.h5：自己训练好的权重。

下载路径：https://blog.csdn.net/Twilight737?spm=1018.2226.3001.5343&type=download

以下4个py文件需要单独运行：

第1步：行annotation.py文件：将CCPD数据集进行划分，生成.txt路径文件存储到model_data文件夹中。

第2步：运行k_means_calculate.py文件：计算生成anchors数值，存储到model_data文件夹中。

第3步：运行train.py文件：加载原始权重，训练YOLOv4模型，并将每轮训练的结果存储进Logs文件夹中。

第4步：运行yolo_predict.py文件：载入训练好的YOLOv4权重，对测试集数据进行检测，检测结果存放入demo文件夹中。



# 算法效果：

YOLOv4每张图片检测耗时323.8ms，精度较高，训练40 epoch左右后val loss降低至0.3附近，效果非常满意。

在测试集中，车牌号检测精度几近100%，效果惊艳。

另外在实验过程中发现，模型过程中的random数据增强非常重要，可大幅提升模型性能，进一步降低val loss。如果不进行random数据增强，训练40 epoch后val loss只能达到1.1附近，且出现瓶颈。但如果使用random数据增强，val loss可降低到0.3附近，相比之前效果好了太多。
