# -*- coding: utf-8 -*-
"""
File run.py
@author:ZhengYuwei
"""
import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras

from configs import FLAGS
from utils.logger import generate_logger
from dataset.file_util import FileUtil
from yolov2.trainer import YOLOv2Trainer
from yolov2.yolov2_decoder import YOLOv2Decoder
from yolov2.yolov2_post_process import YOLOv2PostProcessor

if FLAGS.mode in ('test', 'predict'):
    tf.enable_eager_execution()
if FLAGS.mode == 'train':
    keras.backend.set_learning_phase(True)
else:
    keras.backend.set_learning_phase(False)

keras.backend.set_epsilon(1e-7)
np.random.seed(6)
tf.set_random_seed(800)


def train(yolov2_trainer):
    """ YOLO v2模型训练 """
    logging.info('加载训练数据集：%s', FLAGS.train_label_path)
    train_dataset = FileUtil.get_dataset(FLAGS.train_label_path, FLAGS.train_set_dir,
                                         image_size=FLAGS.input_image_size[0:2],
                                         batch_size=FLAGS.batch_size, is_augment=FLAGS.is_augment, is_test=False)
    yolov2_trainer.train(train_dataset, None)
    logging.info('训练完毕！')


def test(yolov2_trainer, yolov2_decoder, save_path=None):
    """
    YOLO v2模型测试
    :param yolov2_trainer: yolov2检测模型
    :param yolov2_decoder: yolov2模型输出解码器
    :param save_path：测试结果图形报错路径
    """
    logging.info('加载测试数据集：%s', FLAGS.test_label_path)
    test_set = FileUtil.get_dataset(FLAGS.test_label_path, FLAGS.test_set_dir,
                                    image_size=FLAGS.input_image_size[0:2],
                                    batch_size=FLAGS.batch_size, is_augment=False, is_test=True)
    total_test = int(np.ceil(FLAGS.val_set_size / FLAGS.batch_size))
    input_box_size = np.tile(FLAGS.input_image_size[1::-1], [2])  # 网络输入尺度，[W, H, W, H]
    # images为转为[0,1]范围的float32类型的TensorFlow矩阵
    for batch_counter, (images, labels, image_paths) in enumerate(test_set):
        if batch_counter > total_test:
            break
        images, labels, image_paths = np.array(images), np.array(labels), np.array(image_paths)
        predictions = yolov2_trainer.predict(images)
        predictions, predict_boxes = yolov2_decoder.decode(predictions)
        for image, label, image_path, prediction, boxes in zip(images, labels, image_paths,
                                                               np.array(predictions), np.array(predict_boxes)):
            # (k, 8)， 归一化尺度->网络输入尺度的[(left top right bottom iou prob class score) ... ]
            high_score_boxes, max_score = YOLOv2PostProcessor.filter_boxes(prediction, boxes, FLAGS.confidence_thresh)
            nms_boxes = YOLOv2PostProcessor.apply_nms(high_score_boxes, FLAGS.nms_thresh)
            in_boxes = YOLOv2PostProcessor.resize_boxes(nms_boxes, target_size=input_box_size)
            if save_path is not None:
                image_path = os.path.join(save_path, str(os.path.basename(image_path), 'utf-8'))
                YOLOv2PostProcessor.visualize(image, in_boxes, src_box_size=input_box_size, image_path=image_path)
            # TODO 根据预测结果，计算AP，mAP
            # 使用开源库 [Cartucho/mAP](https://github.com/Cartucho/mAP)，真想
    return


def predict(yolov2_trainer, yolov2_decoder, image_paths, save_path):
    """
    YOLO v2模型预测
    :param yolov2_trainer: yolov2检测模型
    :param yolov2_decoder: yolov2模型输出解码器
    :param image_paths: 待预测图片路径列表
    :param save_path：测试结果图形报错路径
    :return:
    """
    import cv2
    logging.info('加载测试数据集：%s', FLAGS.test_label_path)
    input_box_size = np.tile(FLAGS.input_image_size[1::-1], [2])  # 网络输入尺度，(W, H)
    for image_path in image_paths:
        # 读取uint8图片，归一化：[0, 1]的float32 + 原比例resize
        image = tf.constant(cv2.imread(image_path), dtype=tf.uint8)
        image = tf.image.resize_image_with_pad(image, target_height=input_box_size[1], target_width=input_box_size[0],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = np.array(image, dtype=np.float)
        predictions = yolov2_trainer.predict(np.expand_dims(image, axis=0))
        predictions, predict_boxes = yolov2_decoder.decode(predictions)
        prediction, boxes = np.array(predictions)[0], np.array(predict_boxes)[0]
        # (k, 8)， 归一化尺度->网络输入尺度的[(left top right bottom iou prob class score) ... ]
        high_score_boxes = YOLOv2PostProcessor.filter_boxes(prediction, boxes, FLAGS.confidence_thresh)
        nms_boxes = YOLOv2PostProcessor.apply_nms(high_score_boxes, FLAGS.nms_thresh)
        in_boxes = YOLOv2PostProcessor.resize_boxes(nms_boxes, target_size=input_box_size)
        image_path = os.path.join(save_path, os.path.basename(image_path))
        YOLOv2PostProcessor.visualize(image, in_boxes, src_box_size=input_box_size, image_path=image_path)
    return


def run():
    # gpu模式
    if FLAGS.gpu_mode != YOLOv2Trainer.CPU_MODE:
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.visible_gpu
        # tf.device('/gpu:{}'.format(FLAGS.visible_gpu))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # 按需
        sess = tf.Session(config=config)
        """
        # 添加debug：nan或inf过滤器
        from tensorflow.python import debug as tf_debug
        from tensorflow.python.debug.lib.debug_data import InconvertibleTensorProto
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        # nan过滤器
        def has_nan(datum, tensor):
            _ = datum  # Datum metadata is unused in this predicate.
            if isinstance(tensor, InconvertibleTensorProto):
                # Uninitialized tensor doesn't have bad numerical values.
                # Also return False for data types that cannot be represented as numpy
                # arrays.
                return False
            elif (np.issubdtype(tensor.dtype, np.floating) or
                  np.issubdtype(tensor.dtype, np.complex) or
                  np.issubdtype(tensor.dtype, np.integer)):
                return np.any(np.isnan(tensor))
            else:
                return False

        # inf过滤器
        def has_inf(datum, tensor):
            _ = datum  # Datum metadata is unused in this predicate.
            if isinstance(tensor, InconvertibleTensorProto):
                # Uninitialized tensor doesn't have bad numerical values.
                # Also return False for data types that cannot be represented as numpy
                # arrays.
                return False
            elif (np.issubdtype(tensor.dtype, np.floating) or
                  np.issubdtype(tensor.dtype, np.complex) or
                  np.issubdtype(tensor.dtype, np.integer)):
                return np.any(np.isinf(tensor))
            else:
                return False

        # 添加过滤器
        # sess.add_tensor_filter("has_nan", has_nan)
        sess.add_tensor_filter("has_inf", has_inf)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        """
        keras.backend.set_session(sess)
    
    generate_logger(filename=FLAGS.log_path)
    logging.info('TensorFlow version: %s', tf.__version__)  # 1.13.1
    logging.info('Keras version: %s', keras.__version__)  # 2.2.4-tf
    
    yolov2_trainer = YOLOv2Trainer()
    
    # 模型训练
    if FLAGS.mode == 'train':
        train(yolov2_trainer)
    # 多GPU模型，需要先转为单GPU模型，然后再执行测试
    elif FLAGS.mode == 'test' or FLAGS.mode == 'predict':
        # 多GPU模型转换为单GPU模型
        if FLAGS.gpu_num > 1:
            yolov2_trainer.convert_multi2single()
            logging.info('多GPU训练模型转换单GPU运行模型成功，请使用单GPU测试！')
            return
        # 进行测试或预测
        yolov2_decoder = YOLOv2Decoder(grid_size=FLAGS.head_grid_size, class_num=FLAGS.class_num,
                                       anchor_boxes=FLAGS.anchor_boxes)
        save_path = FLAGS.save_path
        if save_path is not None:
            if not os.path.isdir(save_path):
                raise ValueError('测试结果图形报错路径不是文件夹！！')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
        if FLAGS.mode == 'test':
            test(yolov2_trainer, yolov2_decoder, save_path)
            logging.info('测试结束！！！')
        else:
            images_root_path = FLAGS.image_root_path
            if images_root_path is None or not os.path.isdir(save_path) or save_path is None:
                raise ValueError('待预测图形根目录不存在或不是文件夹！！')
            if save_path is None:
                raise ValueError('预测结果图形报错路径不存在！！')
            image_paths = [os.path.join(images_root_path, file_name)
                           for file_name in os.listdir(images_root_path) if file_name.endswith('.jpg')]
            predict(yolov2_trainer, yolov2_decoder, image_paths, save_path)
            logging.info('预测结果！！！')
    # 将模型保存为pb模型
    elif FLAGS.mode == 'save_pb':
        # 保存模型记得注释eager execution
        yolov2_trainer.save_mobile()
    # 将模型保存为服务器pb模型
    elif FLAGS.mode == 'save_serving':
        # 保存模型记得注释eager execution
        yolov2_trainer.save_serving()
    else:
        raise ValueError('Mode Error!')


if __name__ == '__main__':
    run()
