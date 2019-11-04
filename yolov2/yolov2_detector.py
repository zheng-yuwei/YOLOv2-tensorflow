# -*- coding: utf-8 -*-
"""
File yolov2_detector.py
@author:ZhengYuwei
"""
import logging
from tensorflow import keras
from backbone.resnet18 import ResNet18
from backbone.resnet18_v2 import ResNet18_v2
from backbone.resnext import ResNeXt18
from backbone.mixnet18 import MixNet18


class YOLOV2Detector(object):
    """
    检测器，自定义了YOLOv2的head
    """
    BACKBONE_RESNET_18 = 'resnet-18'
    BACKBONE_RESNET_18_V2 = 'resnet-18-v2'
    BACKBONE_RESNEXT_18 = 'resnext-18'
    BACKBONE_MIXNET_18 = 'mixnet-18'
    BACKBONE_MOBILENET = 'mobilenet'
    BACKBONE_MOBILENET_V2 = 'mobilenet-v2'
    BACKBONE_PELEENET = 'peleenet'

    @staticmethod
    def _detection_head(net, output_channel_num, output_head_name):
        """
        YOLOv2的head，上接全卷积网络backbone，输出output_shape个channel的矩阵
        :param net: 全卷积网络backbone
        :param output_channel_num: 检测头channel数，grid_shape = B * （4 + 1 + class_num）
        :param output_head_name: 检测头的名字
        :return: （N * H * W * grid_shape）矩阵
        """
        output = keras.layers.Conv2D(filters=output_channel_num, kernel_size=(1, 1),
                                     kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                     activation=None, use_bias=True, name=output_head_name)(net)
        return output

    @staticmethod
    def build(backbone, input_image_size, output_channel_num, output_head_name):
        """
        构建全卷积网络backbone基础网络的YOLOv2 keras.models.Model对象
        :param backbone: 全卷积网络基础网络，枚举变量 Detector.NetType
        :param input_image_size: 输入尺寸
        :param output_channel_num: 检测头channel数，grid_shape = 5 * （4 + 1 + class_num）
        :param output_head_name: 检测头的名字
        :return: 全卷积网络的YOLOv2 keras.models.Model对象
        """
        if len(input_image_size) != 3:
            raise Exception('模型输入形状必须是3维形式')

        if backbone == YOLOV2Detector.BACKBONE_RESNET_18:
            backbone_func = ResNet18.build
        elif backbone == YOLOV2Detector.BACKBONE_RESNET_18_V2:
            backbone_func = ResNet18_v2.build
        elif backbone == YOLOV2Detector.BACKBONE_RESNEXT_18:
            backbone_func = ResNeXt18.build
        elif backbone == YOLOV2Detector.BACKBONE_MIXNET_18:
            backbone_func = MixNet18.build
        else:
            raise ValueError("没有该类型的基础网络！")

        logging.info('构造YOLOv2模型，基础网络：%s', backbone)
        input_x = keras.layers.Input(shape=input_image_size)
        backbone_model = backbone_func(input_x)
        outputs = YOLOV2Detector._detection_head(backbone_model, output_channel_num, output_head_name)
        model = keras.models.Model(inputs=input_x, outputs=outputs, name=backbone)
        return model


if __name__ == '__main__':
    """
    可视化网络结构，使用plot_model需要先用conda安装GraphViz、pydotplus
    """
    model_names = {
        'resnet-18': YOLOV2Detector.BACKBONE_RESNET_18,
        'resnet-18-v2': YOLOV2Detector.BACKBONE_RESNET_18_V2,
        'resnext-18': YOLOV2Detector.BACKBONE_RESNEXT_18,
        'mixnet-18': YOLOV2Detector.BACKBONE_MIXNET_18
    }
    for key, value in model_names.items():
        test_model = YOLOV2Detector.build(value, input_image_size=(384, 480, 3),
                                          output_channel_num=5 * (4 + 1 + 20), output_head_name='yolov2_head')
        keras.utils.plot_model(test_model, to_file='../images/{}.svg'.format(key), show_shapes=True)
        print('backebone: ', key)
        test_model.summary()
        print('=====================' * 5)
