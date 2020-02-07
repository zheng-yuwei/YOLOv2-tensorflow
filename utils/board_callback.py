# -*- coding: utf-8 -*-
"""
File board_callback.py
@author:ZhengYuwei
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from configs import FLAGS


class MyTensorBoard(keras.callbacks.Callback):
    """
    主要作用包含：
    1. 记录子损失项到tensorboard中；
    2. 记录网络层所有bn层的所有gamma变量的分布到tensorboard中。
    """
    EPOCH_RECORD = False
    BATCH_RECORD = True

    def __init__(self, log_dir='./log', write_graph=True):
        super(MyTensorBoard, self).__init__()
        self.write_graph = write_graph
        self.log_dir = log_dir
        self.sess = None
        self.writer = dict()
        self.steps = 0
        # tensorboard中自定义scalar
        self.metrics = dict()
        self.metrics_keys = None
        # tensorboard中自定义histogram
        self.histograms = dict()
        self.histograms_keys = None

    def set_model(self, model):
        self.model = model

        self._add_sub_loss_scalar()
        self.metrics_keys = set(self.metrics.keys())
        self._add_bn_gamma_histogram()
        self._add_output_iou_histogram()
        self.histograms_keys = set(self.histograms.keys())

        self.sess = keras.backend.get_session()
        # 根目录下放置一个default的日志文件，记录self.metrics以外的scalar
        if self.write_graph:
            self.writer['main'] = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        else:
            self.writer['main'] = tf.summary.FileWriter(self.log_dir)
        # self.metrics内的子损失项、self.histograms分布 放在独立的文件夹里
        for key in self.metrics_keys:
            self.writer[key] = tf.summary.FileWriter(os.path.join(self.log_dir, key))
        for key in self.histograms_keys:
            self.writer[key] = tf.summary.FileWriter(os.path.join(self.log_dir, key))

    def _add_sub_loss_scalar(self):
        """ 添加网络各项损失函数变量 tf.Variable到tensorboard的scalar面板 """
        # 获取各项损失函数变量 tf.Variable
        with tf.variable_scope('loss_detail', reuse=True):
            rectified_coord_loss = tf.get_variable('rectified_coord_loss')
            coord_loss_xy = tf.get_variable('coord_loss_xy')
            coord_loss_wh = tf.get_variable('coord_loss_wh')
            noobj_iou_loss = tf.get_variable('noobj_iou_loss')
            obj_iou_loss = tf.get_variable('obj_iou_loss')
            class_loss = tf.get_variable('class_loss')

        self.metrics.setdefault('rectified_loss', rectified_coord_loss)
        self.metrics.setdefault('xy_loss', coord_loss_xy)
        self.metrics.setdefault('wh_loss', coord_loss_wh)
        self.metrics.setdefault('noobj_iou_loss', noobj_iou_loss)
        self.metrics.setdefault('obj_iou_loss', obj_iou_loss)
        self.metrics.setdefault('class_loss', class_loss)
        return

    def _add_bn_gamma_histogram(self):
        """ 添加网络中各层bn层的gamma变量(tf.Variable(shape=channel_num))到tensorboard的分布面板 """
        bn_gamma = list()
        for variable in self.model.trainable_variables:
            name = variable.name
            if name.find('batch_normalization') != -1 and name.find('gamma') != -1:
                bn_gamma.append(variable)
        self.histograms.setdefault('bn_gamma', tf.concat(bn_gamma, axis=-1))
        return

    def _add_output_iou_histogram(self):
        with tf.variable_scope('loss_detail', reuse=True):
            output_iou = tf.get_variable('output_iou')

        for i in range(FLAGS.box_num):
            self.histograms.setdefault('{}_output_iou'.format(i),
                                       tf.reshape(output_iou[:, :, :, i], [-1]))
        return

    def on_batch_end(self, batch, logs=None):
        if self.BATCH_RECORD:
            logs = logs or {}
            logs.update({'learning_rate': float(keras.backend.get_value(self.model.optimizer.lr))})
            for key, value in self.metrics.items():
                logs.update({key: keras.backend.get_value(value)})
            for key, value in self.histograms.items():
                logs.update({key: keras.backend.eval(value)})
            self._write_logs(logs, self.steps)
            self.steps += 1

    def on_epoch_end(self, epoch, logs=None):
        if self.EPOCH_RECORD:
            logs = logs or {}
            logs.update({'learning_rate': float(keras.backend.get_value(self.model.optimizer.lr))})
            for key, value in self.metrics.items():
                logs.update({key: keras.backend.get_value(value)})
            for key, value in self.histograms.items():
                logs.update({key: keras.backend.eval(value)})
            self._write_logs(logs, self.steps)
            self.steps += 1
        return

    def _write_logs(self, logs, index):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            if name in self.metrics_keys:
                # 所有子loss在tensorboard中都称为loss，好放在同一张图中对比
                self._log_scalar(self.writer[name], 'loss', value, index)
            elif name in self.histograms_keys:
                self._log_histogram(self.writer[name], name, value, index)
            else:
                self._log_scalar(self.writer['main'], name, value, index)

        for writer in self.writer.values():
            writer.flush()

    @staticmethod
    def _log_scalar(writer, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        writer.add_summary(summary, step)

    @staticmethod
    def _log_histogram(writer, tag, values, step, bins=1000):
        """
        Logs the histogram of a list/vector of values.
        ref: https://gist.github.com/FireJohnny/1f82b5f7a3eabbdc7aacdb967fe1b557
        """
        # Convert to a numpy array
        values = np.array(values)
        values = np.sort(values)[:int(len(values)*99/100)]
        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        writer.add_summary(summary, step)

    def on_train_end(self, logs=None):
        for writer in self.writer.values():
            writer.close()
