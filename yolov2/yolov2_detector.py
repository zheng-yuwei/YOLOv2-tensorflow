# -*- coding: utf-8 -*-
"""
File yolov2_trainer.py
@author:ZhengYuwei
"""
import logging
import tensorflow as tf
from tensorflow import keras
from backbone.resnet18 import ResNet18
from backbone.resnet18_v2 import ResNet18_v2
from backbone.resnext import ResNeXt18
from backbone.mixnet18 import MixNet18
from backbone.mobilenet_v2 import MobileNetV2

from configs import FLAGS, Backbone


class SampleFreeBias(keras.initializers.Initializer):
    """初始化并生成一个指定维度为预定义常数的tensor，思想来源于sample-free的论文：
    sigmoid(x) = N_f / (N * C) => x = -log((N*C)/N_f - 1)
    Is Sampling Heuristics Necessary in Training Deep Object Detectors?
    (https://arxiv.org/pdf/1909.04868.pdf)
    """

    def __init__(self, dtype=tf.dtypes.float32, constant=0., anchor_num=5):
        self.dtype = tf.dtypes.as_dtype(dtype)
        self.constant = constant
        self.anchor_num = anchor_num

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        new_shape = shape[0] // self.anchor_num
        biases = list()
        for c in range(self.anchor_num):
            biases.extend([tf.zeros(4, dtype), tf.constant(self.constant, dtype, shape=(1,)),
                           tf.zeros(new_shape - 5, dtype)])
        return tf.concat(biases, axis=-1)

    def get_config(self):
        return {"dtype": self.dtype.name}


class YOLOv2Detector(object):
    """
    检测器，自定义了YOLOv2的head
    """
    BACKBONE_TYPE = {
        Backbone.resnet18: ResNet18,
        Backbone.resnet18_v2: ResNet18_v2,
        Backbone.resnext18: ResNeXt18,
        Backbone.mobilenet_v2: MobileNetV2,
        Backbone.mixnet18: MixNet18
    }

    @classmethod
    def _detection_head(cls, net, head_channel_num, head_name):
        """
        YOLOv2的head，上接全卷积网络backbone，输出output_shape个channel的矩阵
        :param net: 全卷积网络backbone
        :param head_channel_num: 检测头channel数，grid_shape = B * （4 + 1 + class_num）
        :param head_name: 检测头的名字
        :return: （N * H * W * grid_shape）矩阵
        """
        initializer = 'zeros'
        if FLAGS.is_sample_free:
            initializer = SampleFreeBias(constant=FLAGS.bias_constant, anchor_num=FLAGS.box_num)
        output = keras.layers.Conv2D(filters=head_channel_num, kernel_size=(1, 1),
                                     kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                     bias_initializer=initializer,
                                     activation=None, use_bias=True, name=head_name)(net)
        return output

    @classmethod
    def build(cls, backbone_name, input_image_size, head_channel_num, head_name):
        """
        构建全卷积网络backbone基础网络的YOLOv2 keras.models.Model对象
        :param backbone_name: 全卷积网络基础网络，枚举变量 Detector.NetType
        :param input_image_size: 输入尺寸
        :param head_channel_num: 检测头channel数，grid_shape = 5 * （4 + 1 + class_num）
        :param head_name: 检测头的名字
        :return: 全卷积网络的YOLOv2 keras.models.Model对象
        """
        if len(input_image_size) != 3:
            raise Exception('模型输入形状必须是3维形式')
        
        if backbone_name in cls.BACKBONE_TYPE.keys():
            backbone = cls.BACKBONE_TYPE[backbone_name]
        else:
            raise ValueError("没有该类型的基础网络！")

        logging.info('构造YOLOv2模型，基础网络：%s', backbone_name)
        input_x = keras.layers.Input(shape=input_image_size)
        backbone_output = backbone.build(input_x)
        outputs = cls._detection_head(backbone_output, head_channel_num, head_name)
        model = keras.models.Model(inputs=input_x, outputs=outputs, name=backbone_name)
        return model


if __name__ == '__main__':
    """
    可视化网络结构，使用plot_model需要先用conda安装GraphViz、pydotplus
    """
    for test_backbone_name in YOLOv2Detector.BACKBONE_TYPE.keys():
        test_model = YOLOv2Detector.build(test_backbone_name, input_image_size=(384, 480, 3),
                                          head_channel_num=5 * (4 + 1 + 20), head_name='yolov2_head')
        keras.utils.plot_model(test_model, to_file='../images/{}.svg'.format(test_backbone_name), show_shapes=True)
        print('backbone: ', test_backbone_name)
        test_model.summary()
        print('=====================' * 5)
