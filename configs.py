# -*- coding: utf-8 -*-
"""
File configs.py
@author:ZhengYuwei
"""
import datetime
import numpy as np
from easydict import EasyDict
from yolov2.yolov2_detector import YOLOV2Detector


def lr_func(epoch):
    # step_epoch = [10, 20, 30, 40, 50, 60, 70, 80]
    # step_lr = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0] # 0.0001
    step_epoch = [20, 60, 80, 220, 260, 280, 300]
    step_lr = [0.00001, 0.001, 0.0001, 0.001, 0.0001, 0.00001, 0.000001]
    i = 0
    while i < len(step_epoch) and epoch > step_epoch[i]:
        i += 1
    return step_lr[i]


FLAGS = EasyDict()

# 数据集
FLAGS.train_set_dir = 'dataset/test_sample/images'  # '/cache/zhengyuwei/new_plate_images'
FLAGS.train_label_path = 'dataset/test_sample/label.txt'  # '/cache/zhengyuwei/new_plate_images/train.txt'
FLAGS.test_set_dir = 'dataset/test_sample/images'  # '/cache/zhengyuwei/new_plate_images'
FLAGS.test_label_path = 'dataset/test_sample/label.txt'  # '/cache/zhengyuwei/new_plate_images/val.txt'
# 模型权重的L2正则化权重直接写在对应模型的骨干网络定义文件中
FLAGS.input_image_size = np.array([384, 480, 3], dtype=np.int)  # [(]H, W, C]
FLAGS.anchor_boxes = [(0.1006, 0.1073), (0.453, 0.5203), (0.3096, 0.1985), (0.1527, 0.3528), (0.7907, 0.7889)]  # [W, H]
FLAGS.class_num = 0
FLAGS.box_num = len(FLAGS.anchor_boxes)
FLAGS.box_len = 4 + 1 + FLAGS.class_num
FLAGS.output_channel_num = FLAGS.box_num * FLAGS.box_len
FLAGS.output_grid_size = np.divide(FLAGS.input_image_size[0:2], 32).astype(np.int)  # [H, W]
FLAGS.output_head_name = 'yolov2_head'
FLAGS.iou_thresh = 0.7  # 大于该IOU阈值，不计算该anchor的背景IOU误差
FLAGS.loss_weights = [50, 100, 0.05, 10, 10]  # 不同损失项的权：[coord_xy, coord_wh, noobj, obj, cls_prob]
# 训练参数
FLAGS.train_set_size = 14  # 160108
FLAGS.val_set_size = 14  # 35935
FLAGS.batch_size = 5  # 3079
# 但你已经有预训练模型时，给rectified_coord_num赋值为0即可
FLAGS.rectified_coord_num = 915  # 前期给坐标做矫正损失的图片数，源代码 12800，train-from-scratch需要用
FLAGS.rectified_loss_weight = 1.0  # 前期矫正坐标的损失的权重，源代码 0.01，具体可调，太大的话coord_loss_wh会跟着爆炸
FLAGS.epoch = 300
FLAGS.init_lr = 0.0002  # nadam推荐使用值
# 训练参数
FLAGS.mode = 'test'  # train, test, predict, save_pb, save_serving
FLAGS.model_backbone = YOLOV2Detector.BACKBONE_RESNET_18
FLAGS.optimizer = 'radam'  # sgdm, adam, radam
FLAGS.is_augment = True
FLAGS.is_label_smoothing = False
FLAGS.is_focal_loss = False
FLAGS.is_gradient_harmonized = False
FLAGS.type = FLAGS.model_backbone + '-' + FLAGS.optimizer
FLAGS.type += ('-aug' if FLAGS.is_augment else '')
FLAGS.type += ('-smooth' if FLAGS.is_label_smoothing else '')
FLAGS.type += ('-focal' if FLAGS.is_focal_loss else '')
FLAGS.type += ('-ghm' if FLAGS.is_gradient_harmonized else '')
FLAGS.log_path = 'logs/log-{}.txt'.format(FLAGS.type)
# 训练参数
FLAGS.steps_per_epoch = int(np.ceil(FLAGS.train_set_size / FLAGS.batch_size))
FLAGS.validation_steps = int(np.ceil(FLAGS.val_set_size / FLAGS.batch_size))
# callback的参数
FLAGS.ckpt_period = 50  # 模型保存
FLAGS.stop_patience = 500  # early stop
FLAGS.stop_min_delta = 0.0001
FLAGS.lr_func = lr_func  # 学习率更新函数
# tensorboard日志保存目录
FLAGS.root_path = ''  # /cache/zhengyuwei/license-plate-recognition/
FLAGS.tensorboard_dir = FLAGS.root_path + 'logs/' + \
                        'lpr-{}-{}'.format(FLAGS.type, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
# 模型保存
FLAGS.checkpoint_path = FLAGS.root_path + 'models/{}/'.format(FLAGS.type)
FLAGS.checkpoint_name = 'lp-recognition-{}'.format(FLAGS.type) + '-{epoch: 3d}-{loss: .5f}.ckpt'
FLAGS.serving_model_dir = FLAGS.root_path + 'models/serving'
FLAGS.pb_model_dir = FLAGS.root_path + 'models/pb'

# 测试参数
FLAGS.confidence_thresh = 0.5  # 基础置信度
FLAGS.nms_thresh = 0.4  # nms阈值
FLAGS.save_path = 'dataset/test_result/'  # 测试结果图形报错路径
FLAGS.image_root_path = None  # 预测图片的根目录
# 训练gpu
FLAGS.gpu_mode = 'gpu'
FLAGS.gpu_num = 1
FLAGS.visible_gpu = '0'  # ','.join([str(_) for _ in range(FLAGS.gpu_num)])
