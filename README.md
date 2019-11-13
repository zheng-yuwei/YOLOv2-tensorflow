# YOLOv2-tensorflow

By 郑煜伟

基于tf.keras，实现YOLOv2模型。
## 本项目相比其他YOLO v2项目的特色

与TensorFlow版本的YOLO v2项目相比，**最大程度遵从原始论文、原始Darknet框架的实现&**
（可以说解决了逻辑bug吧，具体可查看`1_learning_note/implementation_process.ipynb`）；

与原版Darknet、Caffe实现相比，train-from-scratch的预测框校正功能可根据实际情况进行开启/关闭；

与所有YOLO v2项目相比：
1. 使用tf.data.Dataset读取数据，tf.keras构造模型，简单易懂，同时易于多GPU训练、模型转换等操作；
2. 全中文详细代码注释，算法理解等说明。

## 如何使用

取coco数据集中的20张图片做训练，测试效果如下，更多结果可查看`dataset/test_result*`。

![测试结果](./dataset/test_result/000004.jpg)

### 快速上手
1. 制作数据集`label.txt`，一行为`image_path x0 y0 w0 h0 cls0 x1 y1 x1 h1 cls1 ...`，
其中`xywh`为待检测目标的bounding box中心点坐标和宽高相对于原图的比例（归一化了），`cls`为类别；
1. 实际用自己的数据训练时，可能需要执行以下`utils/check_label_file.py`，确保标签文件中的图片真实可用；
1. 修改并运行`utils/anchors/kmeans_anchors.py`，聚类预定义anchors；
1. run.py同目录下新建 `logs`文件夹，存放日志文件；训练完毕会出现`models`文件夹，存放模型；
1. 查看`configs.py`并进行修改，此为参数配置文件；
1. 执行`python run.py`，会根据配置文件`configs.py`进行训练/测试/模型转换等（需要注意我设置了随机种子）。

![anchor聚类图](./images/k-menas++anchors.png)

![不同聚类中心下，待检测目标与归属anchor的IOU-样本比例的ROC曲线](./images/IOU-Ratio-curve.png)

### 学习掌握
1. 先看`README.md`;
2. 再看`1_learning_note`下的note；
3. 看`multi_label`下的`trainer.py`里的`__init__`函数，把整体模型串起来；
4. 看`run.py`文件，结合着看`configs.py`。

## 目录结构

- `A_learning_notes`: README后，**先查看本部分**了解本项目大致结构；
- `backbone`: 模型的骨干网络脚本，`basic_backbone.py`包含了基类`BasicBackbone`，
实现了5个类型的骨干网络：`resnet-18`, `resnet-18-v2`, `mobilenet-v2`, `mixnet-18`, `resnext-18`；
其中，前三个网络基本遵照原始网络结构，后两个是借鉴了对应网络的思想，在`resnet-18`基础上改写；
- `dataset`: 数据集构造脚本；
    - `dataset_util.py`: 使用tf.image API进行图像数据增强，然后用tf.data进行数据集构建；
    - `file_util.py`: 以txt标签文件的形式，构造tf.data数据集用于训练；
- `images`: 项目图片；
- `logs`: 存放训练过程中的日志文件和tensorboard文件（当前可能不存在）；
- `models`: 存放训练好的模型文件（当前可能不存在）；
- `utils`: 一些工具脚本；
    - `anchors`: 通过k-means聚类计算得到预定义anchors；
    - `check_label_file.py`: 在训练前检查训练集，确保标签文件中的图片真实可用；
    - `logger.py`：构造文件和控制台日志句柄；
    - `logger_callback.py`: 日志打印的keras回调函数；
    - `radam.py`: RAdam算法的tf.keras优化器实现；
- `yolov2`: yolov2模型构建脚本；
    - `train.py`: 模型训练接口，集成模型构建/编译/训练/debug/预测、数据集构建等功能；
    - `yolov2_decoder.py`: 对YOLO v2模型的预测输出进行解码；
    - `yolov2_trainer.py`: 构造YOLO v2检测器模型；
    - `yolov2_loss.py`: YOLO v2的损失函数；
- `configs.py`: 配置文件；
- `run.py`: 启动脚本；


## 代码库特别说明

### 标签文件格式说明

标签文件格式内容为：
```
image_path x0 y0 w0 h0 cls0 ...
```
其中，`image_path`是图片相对路径，会拼接上`configs.py`中的`FLAGS.train_set_dir`（测试的话则是`FLAGS.test_set_dir`）；
`x0 y0 w0 h0`是归一化后的待检测物品中心点坐标、宽高，归一化也就是 实际尺寸/图片尺寸；
`cls0`是图片类别，即使是单类别且不计算类别损失，该位也必须存在（可以任意值）。
前者是多类别的目标检测，后者主要是单类别的目标检测。
后续省略号表示多个待检测对象的标签`x0 y0 w0 h0 cls0`。

## 算法说明

[YOLOv2](https://zheng-yuwei.github.io/2018/10/03/4_YOLOv2/)

## TODO
- [] 多尺度输入;
- [] mixup;
- [x] focal loss;
- [] GHM损失函数;
- [] GIOU;
- [x] TIOU-Recall;
- [] Guassian YOLO;
- [] 模型测试，计算mAP；
