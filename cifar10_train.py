# -*- coding:utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10


# FLAGS用于传递 tf.app.run( ) 所需的参数
FLAGS = tf.app.flags.FLAGS
# 定义训练集路径
tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
# 训练轮数
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
# 是否打印设备分配日志，找指令和张量被分配到哪个设备
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
# 打印日志频率
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        # as_default()用于保存变量信息

        # 记录全局更新轮数
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        # 获取数据集和标签
        images, labels = cifar10.distorted_inputs()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        # 构建一个图形，用于计算推理模型中的logits的预测值
        # 使用CNN，经过卷积和池化层处理。返回处理后结果
        logits = cifar10.inference(images)

        # Calculate loss.
        # 求误差值
        loss = cifar10.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        # 训练建立好的模型
        train_op = cifar10.train(loss, global_step) #训练模型，内含学习率，损失函数，梯度衰减

        class _LoggerHook(tf.train.SessionRunHook): #tf.train.SessionRunHook类似于 Session 的一个处理初始化, 恢复和 hooks 的功能
            """Logs loss and runtime."""

            def begin(self):        #初始化（step和时间）
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):   #每运行一定轮数，打印出一些信息（运行时间，次数，loss值，每秒次数，每秒batch数）
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,             #路径
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps), #总轮数
                   tf.train.NanTensorHook(loss),        #监控loss，如果loss为NaN则停止训练
                   _LoggerHook()],                      #初始化
            config=tf.ConfigProto(                  #记录设备指派情况
                log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():       #循环运行run()
                mon_sess.run(train_op)

def main(argv=None):  # pylint: disable=unused-argument
    # 下载和解压缩，函数位于cifar10.py
    # 先查找本级目录是否存在数据集文件，若不存在，则在线下载
    cifar10.maybe_download_and_extract()

# 判断目录或文件是否存在，如果是，则递归删除train_dir所有目录及其文件
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
# 以递归方式建立train_dir的父目录及其子目录
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
