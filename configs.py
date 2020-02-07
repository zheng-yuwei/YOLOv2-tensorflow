# -*- coding: utf-8 -*-
"""
File configs.py
@author:ZhengYuwei
"""
import datetime
import numpy as np
from easydict import EasyDict


class Backbone(object):
    resnet18 = 'resnet-18'
    resnet18_v2 = 'resnet-18-v2'
    resnext18 = 'resnext-18'
    mixnet18 = 'mixnet-18'
    mobilenet_v2 = 'mobilenet-v2'


FLAGS = EasyDict()

# check_*是在项目前期定学习率时用于测试的，后续调整train_*学习率就行
FLAGS.check_step_epoch = np.array([2, 4, 6, 8, 10, 12, 14], np.int)
FLAGS.check_step_lr = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1., 10.0], dtype=np.float) * 1e-3
FLAGS.train_step_epoch = np.array([20, 60, 80, 220, 260, 280, 300], np.int)
FLAGS.train_step_lr = np.array([0.01, 1., 0.1, 1., 0.1, 0.01, 0.001], dtype=np.float) * 1e-3
# 训练时期或是调整学习率时期
FLAGS.step_epoch = FLAGS.train_step_epoch
FLAGS.step_lr = FLAGS.train_step_lr


def lr_func(epoch):
    i = 0
    while i < len(FLAGS.step_epoch) and epoch > FLAGS.step_epoch[i]:
        i += 1
    return FLAGS.step_lr[i]


FLAGS.epoch = 300  # 训练迭代epoch
FLAGS.init_lr = 1e-3  # 优化器初始学习率
# callback的参数
# ModelCheckpoint
FLAGS.ckpt_period = 10  # 模型保存
# EarlyStopping
FLAGS.stop_patience = 50  # early stop
FLAGS.stop_min_delta = 1e-7
# LearningRate
FLAGS.lr_func = lr_func  # 学习率更新函数
FLAGS.lr_self_define = True
FLAGS.lr_decay_rate = 0.2
FLAGS.lr_warmup_epoch = 3
FLAGS.lr_warmup_rate = 1e-2
FLAGS.lr_min_delta = 1e-3
FLAGS.lr_patience = 2
FLAGS.lr_minimum = 1e-8

# 数据集
FLAGS.train_set_dir = 'dataset/test_sample/images'
FLAGS.train_label_path = 'dataset/test_sample/label.txt'
FLAGS.test_set_dir = 'dataset/test_sample/images'
FLAGS.test_label_path = 'dataset/test_sample/label.txt'
# 测试参数
FLAGS.save_path = 'dataset/test_result/'  # 测试结果图形报错路径
FLAGS.image_root_path = 'dataset/val_neg_images'  # 预测图片的根目录
FLAGS.confidence_thresh = 0.8  # 基础置信度
FLAGS.nms_thresh = 0.55  # nms阈值
# 模型权重的L2正则化权重直接写在对应模型的骨干网络定义文件中
FLAGS.input_image_size = np.array([384, 480, 3], dtype=np.int)  # [H, W, C]
FLAGS.anchor_boxes = [(0.1006, 0.1073), (0.453, 0.5203), (0.3096, 0.1985), (0.1527, 0.3528), (0.7907, 0.7889)]  # [W, H]
FLAGS.class_num = 0
FLAGS.box_num = len(FLAGS.anchor_boxes)
FLAGS.box_len = 4 + 1 + FLAGS.class_num
FLAGS.head_channel_num = FLAGS.box_num * FLAGS.box_len
FLAGS.head_grid_size = np.divide(FLAGS.input_image_size[0:2], 32).astype(np.int)  # [H, W]
FLAGS.head_name = 'yolov2_head'
FLAGS.iou_thresh = 0.7  # 大于该IOU阈值，不计算该anchor的背景IOU误差
FLAGS.loss_weights = [0.5, 5, 0.05, 2, 2]  # 不同损失项的权：[coord_xy, coord_wh, noobj, obj, cls_prob]
# 训练参数
FLAGS.train_set_size = 20
FLAGS.val_set_size = 20
FLAGS.batch_size = 6
# 若你已经有预训练模型，给rectified_coord_num赋值为-1即可
FLAGS.rectified_coord_num = 100  # 前期给坐标做矫正损失的图片数，源代码 12800，train-from-scratch需要用
FLAGS.rectified_loss_weight = 0.1  # 前期矫正坐标的损失的权重，源代码 0.01，太大的话coord_loss_wh会跟着爆炸
# 训练参数
FLAGS.mode = 'train'  # train, test, predict, save_pb, save_serving
FLAGS.model_backbone = Backbone.resnet18
FLAGS.optimizer = 'radam'  # sgdm, adam, radam
FLAGS.is_augment = False
FLAGS.is_label_smoothing = False
FLAGS.is_sample_free = False
# 这个值需要根据实际计算 Is Sampling Heuristics Necessary in Training Deep Object Detectors?
FLAGS.bias_constant = -np.log(FLAGS.box_num * FLAGS.input_image_size[0] * FLAGS.input_image_size[1] - 1)
FLAGS.is_focal_loss = False
FLAGS.focal_alpha = 1.0
FLAGS.focal_gamma = 2.0
FLAGS.is_gradient_harmonized = False
FLAGS.is_tiou_recall = False
FLAGS.type = FLAGS.model_backbone + '-' + FLAGS.optimizer
FLAGS.type += ('-aug' if FLAGS.is_augment else '')
FLAGS.type += ('-smooth' if FLAGS.is_label_smoothing else '')
FLAGS.type += ('-smpfree' if FLAGS.is_sample_free else '')
FLAGS.type += ('-focal' if FLAGS.is_focal_loss else '')
FLAGS.type += ('-ghm' if FLAGS.is_gradient_harmonized else '')
FLAGS.type += ('-TIOU' if FLAGS.is_tiou_recall else '')
FLAGS.log_path = 'logs/log-{}.txt'.format(FLAGS.type)
# 训练参数
FLAGS.steps_per_epoch = int(np.ceil(FLAGS.train_set_size / FLAGS.batch_size))
FLAGS.validation_steps = int(np.ceil(FLAGS.val_set_size / FLAGS.batch_size))
# tensorboard日志保存目录
FLAGS.root_path = ''  # /cache/zhengyuwei/license-plate-recognition/
FLAGS.tensorboard_dir = FLAGS.root_path + 'logs/' + \
                        'lpr-{}-{}'.format(FLAGS.type, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
# 模型保存
FLAGS.checkpoint_path = FLAGS.root_path + 'models/{}/'.format(FLAGS.type)
FLAGS.checkpoint_name = 'lp-recognition-{}'.format(FLAGS.type) + '-{epoch: 3d}-{loss: .5f}.ckpt'
FLAGS.serving_model_dir = FLAGS.root_path + 'models/serving'
FLAGS.pb_model_dir = FLAGS.root_path + 'models/pb'

# 训练gpu
FLAGS.gpu_mode = 'gpu'
FLAGS.gpu_num = 1
FLAGS.visible_gpu = '0'  # ','.join([str(_) for _ in range(FLAGS.gpu_num)])
