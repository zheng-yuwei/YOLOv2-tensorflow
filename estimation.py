# -*- coding: utf-8 -*-
"""模型预测"""
import os
import cv2
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc

from configs import FLAGS
from yolov2.yolov2_post_process import YOLOv2PostProcessor


class Estimation(object):

    CATEGORY = ['w/o text', 'with text']

    @staticmethod
    def parse(mode):
        if mode == 'test':
            file_path = FLAGS.test_label_path
            root_dir = FLAGS.test_set_dir
        else:
            file_path = FLAGS.val_label_path
            root_dir = FLAGS.val_set_dir

        image_paths = list()
        with open(file_path, 'r') as label_file:
            for line in label_file:
                line = line.strip().split(',')
                image_name = line[0]
                label = int(line[1])
                image_paths.append((os.path.join(root_dir, image_name), label))
        return image_paths

    @staticmethod
    def eval(yolov2_trainer, yolov2_decoder, mode):
        """
        YOLO v2模型预测
        :param yolov2_trainer: yolov2检测模型
        :param yolov2_decoder: yolov2模型输出解码器
        :param mode: 评估模式，test 或 val
        :return:
        """
        logging.info('加载数据集：%s', mode)
        input_box_size = np.tile(FLAGS.input_image_size[1::-1], [2])  # 网络输入尺度，(W, H)
        image_paths = Estimation.parse(mode)
        labels = list()
        preds = list()
        scores = list()
        for image_path, label in image_paths:
            # 预处理
            src_image = cv2.imread(image_path)
            image = tf.constant(src_image, dtype=tf.uint8)
            image = tf.image.resize_image_with_pad(image, target_height=input_box_size[1], target_width=input_box_size[0],
                                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = np.array(image, dtype=np.float)
            # 预测
            predictions = yolov2_trainer.predict(np.expand_dims(image, axis=0))
            # 后处理
            predictions, predict_boxes = yolov2_decoder.decode(predictions)
            prediction, boxes = np.array(predictions)[0], np.array(predict_boxes)[0]
            # (k, 8)， 归一化尺度->网络输入尺度的[(left top right bottom iou prob class score) ... ]
            high_score_boxes, max_score = YOLOv2PostProcessor.filter_boxes(prediction, boxes, FLAGS.confidence_thresh)
            nms_boxes = YOLOv2PostProcessor.apply_nms(high_score_boxes, FLAGS.nms_thresh)
            pred = 0
            if nms_boxes:
                pred = 1

            # in_boxes = YOLOv2PostProcessor.resize_boxes(nms_boxes, target_size=input_box_size)
            # if pred == 1 and label == 0:
            #     image_path = os.path.join('dataset/test_result', os.path.basename(image_path))
            #     YOLOv2PostProcessor.visualize(src_image.astype(np.float32)/255,
            #                                   in_boxes,
            #                                   src_box_size=input_box_size,
            #                                   image_path=image_path)
            preds.append(pred)
            labels.append(label)
            scores.append(max_score)

        logging.info('\n' + classification_report(labels, preds, target_names=Estimation.CATEGORY))
        Estimation.viz_roc(labels, scores, mode)
        return

    @staticmethod
    def viz_roc(labels, scores, mode):
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr,
                 label='ROC curve (area = {0:0.2f})'.format(roc_auc),
                 color='deeppink', linestyle=':', linewidth=4)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('CategoryLite with {} ROC curve and AUC'.format(mode))
        plt.legend(loc="lower right")
        plt.grid(axis='y')
        plt.grid(axis='x')
        plt.show()
