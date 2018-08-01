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

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
# 处理此大小的图像。 请注意，这与原始CIFAR图像大小32 x 32不同。
# 如果更改此数字，则整个模型体系结构将发生变化，任何模型都需要重新训练。
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
# 描述CIFAR-10数据集的全局常量。
NUM_CLASSES = 10 # 10类
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000 # 训练量 
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000 # 测试量


def read_cifar10(filename_queue):
  """Reads and parses examples from CIFAR10 data files.
     从CIFAR10数据文件中读取和解析示例。

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.


  Args:
    filename_queue: A queue of strings with the filenames to read from.
                    具有要读取的文件名的字符串队列。
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  返回：
     表示单个示例的对象，包含以下字段：
       height：结果中的行数（32）
       width：结果中的列数（32）
       depth：结果中的颜色通道数（3）
       key：标量字符串Tensor，描述此示例的文件名和记录号。
       label：一个int32 Tensor，标签范围为0..9。
       uint8image：a [高度，宽度，深度] uint8张量图像数据
  """
  # result 记录存储要输出的信息。
  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  # CIFAR-10数据集中图像的尺寸。
  # 分类结果的长度
  label_bytes = 1       
  # （2 for CIFAR-100   CIFAR-100长度为2）
  result.height = 32
  result.width = 32
  # 3位表示rgb颜色（0-255,0-255,0-255）
  result.depth = 3
  # 图片长度
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  # 单个记录的总长度=分类结果长度+图片长度
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  # 根据record_bytes定义一个reader对象
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  # 从对应filename_queue读取信息
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  # 把value从字符串转换成uint8，
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  # tf.cast：用于改变某个张量的数据类型
  # 第一位代表lable-图片的正确分类结果，从uint8转换为int32类型
  result.label = tf.cast(
      # 从0到label_bytes的左闭右开区间，对record_bytes进行切片处理，默认步长每个维度为1
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  # 标签后面的剩余字节代表图像，我们将其从[颜色*高度*宽度]重新整形为[颜色，高度，宽度]。
  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
      [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  # 格式转换，从[颜色,高度,宽度]--》[高度,宽度,颜色]
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.
     构建一个排列后的一组图像和标签。
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  ARGS：
     image：type.float32的[高度，宽度，3]的3-D张量。
     label：1-D Tensor of type.int32
     min_queue_examples：int32，要保留的最小样本数在提供批量示例的队列中。
     batch_size：每批图像数。
     shuffle：指示是否使用混洗队列的布尔值。

  返回：
     images：Images. [batch_size，height，width，3]大小的4D张量。
     labels：Labels. [batch_size]大小的1D张量。
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  # 创建一个对示例进行混洗的队列，然后从示例队列中读取“batch_size”图像+标签。
  # 线程数
  num_preprocess_threads = 16

  if shuffle:
    # 从[image, label]利用num_threads个线程读取batch_size行
    images, label_batch = tf.train.shuffle_batch(
        [image, label], # 入队的张量列表
        batch_size=batch_size,      # 表示进行一次批处理的tensors数量.
        num_threads=num_preprocess_threads, # 使用num_threads个线程在tensor_list中读取文件
        capacity=min_queue_examples + 3 * batch_size, # 容量:一个整数,队列中的最大的元素数. 
        min_after_dequeue=min_queue_examples) # 当一次出列操作完成后,队列中元素的最小数量,往往用于定义元素的混合级别.
                                              # 一定要保证这参数小于capacity参数的值
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)
  # label_batch：样本标签
  # Display the training images in the visualizer.
  # 在可视化工具中显示训练图像。
  tf.summary.image('images', images)
  # tf.reshape(label_batch, [batch_size]) 将label_batch变换为参数[batch_size]的形式，矩阵变换
  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.
     使用Reader ops为CIFAR培训构建扭曲的输入。
  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

   ARGS：
     data_dir：CIFAR-10数据目录的路径。
     batch_size：每批图像数。

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  返回：
     images：Images. [batch_size，height，width，3]大小的4D张量。
     labels：Labels. [batch_size]大小的1D张量。
  """
  # os.path.join()将多个路径组合后返回
  filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
               #for i in xrange(1, 6)]
               for i in range(1, 6)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  # 创建一个要读取的文件名的队列。
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  # 从文件名队列中的文件中读取示例。
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for training the network. Note the many random
  # distortions applied to the image.
  # 用于训练网络的图像处理，注意使用图像的随机失真
  # Randomly crop a [height, width] section of the image.
  # 随机裁剪图片
  distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

  # Randomly flip the image horizontally.
  # 随机旋转图片
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  # 因为这些操作都是不可交换的，所以考虑随机化它们的操作顺序
  # 随机亮度变换
  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
  # 随机对比度变换
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)
  # Subtract off the mean and divide by the variance of the pixels.
  # 减去均值并除以像素的方差。
  # 标准化
  float_image = tf.image.per_image_standardization(distorted_image)

  # Set the shapes of tensors.
  # 设置张量的型
  float_image.set_shape([height, width, 3])
  read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  # 确保随机性
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print('Filling queue with %d CIFAR images before starting to train. '
        'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  # 通过构建示例队列来生成一批图像和标签。
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)


def inputs(eval_data, data_dir, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.
     使用Reader ops为CIFAR评估构造输入。
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  ARGS：
     eval_data：bool，表示是否应该使用train或eval数据集。
     data_dir：CIFAR-10数据目录的路径。
     batch_size：每批图像数。
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # train数据集
  if not eval_data:
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  # eval数据集
  else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  # 创建一个要读取的文件名的队列。
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  # 从文件名队列中的文件中读取示例。
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for evaluation.
  # 用于评估的图像处理
  # Crop the central [height, width] of the image.
  # 裁剪图像的中央
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         height, width)

  # Subtract off the mean and divide by the variance of the pixels.
  # 减去均值并除以像素的方差。
  # 标准化
  float_image = tf.image.per_image_standardization(resized_image)

  # Set the shapes of tensors.
  # 设定张量的型
  float_image.set_shape([height, width, 3])
  read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  # 确保随机性
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  # 通过构建示例队列来生成一批图像和标签。
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)
