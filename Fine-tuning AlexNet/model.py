import tensorflow as tf
import numpy as np


class AlexNet(object):

  def __init__(self, x, keep_prob, num_classes, skip_layer, fc, conv, additional,
               weights_path = 'DEFAULT'):
    """
    Inputs:
    - x: tf.placeholder, for the input images
    - keep_prob: tf.placeholder, for the dropout rate
    - num_classes: int, number of classes of the new dataset
    - skip_layer: list of strings, names of the layers you want to reinitialize
    - weights_path: path string, path to the pretrained weights,
                    (if bvlc_alexnet.npy is not in the same folder)
    """
    # Parse input arguments
    self.x = x
    self.num_classes = num_classes
    self.keep_prob = keep_prob
    self.skip_layer = skip_layer
    self.fc = fc
    self.conv = conv
    self.additional = additional

    if weights_path == 'DEFAULT':
      self.weights_path = 'bvlc_alexnet.npy'
    else:
      self.weights_path = weights_path

    # Call the create function to build the computational graph of AlexNet
    self.create()

  def create(self):

      conv1 = conv(self.x, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
      pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
      out = lrn(pool1, 2, 2e-05, 0.75, name='norm1')

      if self.conv >= 2:
          conv2 = conv(out, 5, 5, 256, 1, 1, groups=2, name='conv2')
          pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
          out = lrn(pool2, 2, 2e-05, 0.75, name='norm2')

      else:
          self.skip_layer.append('conv2') # Prevent trained weights to be loaded

      if self.conv >= 3:
          out = conv(out, 3, 3, 384, 1, 1, name='conv3')
      else:
          self.skip_layer.append('conv3')

      if self.conv >= 4:
          out = conv(out, 3, 3, 384, 1, 1, groups=2, name='conv4')
      else:
          self.skip_layer.append('conv4')

      if self.conv == 5:
          conv5 = conv(out, 3, 3, 256, 1, 1, groups=2, name='conv5')
          out = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
      else:
          self.skip_layer.append('conv5')

      if self.additional and self.conv == 5:
          conv6 = conv(out, 1, 1, 128, 1, 1, name='conv6')
          conv7 = conv(conv6, 3, 3, 128, 1, 1, name='conv7')
          out = conv(conv7, 1, 1, 256, 1, 1, name='conv8')

      shape = out.get_shape()
      flattened_shape = int(shape[1] * shape[2] * shape[3])

      flattened = tf.reshape(out, [-1, flattened_shape])
      
      if self.fc != 4096: self.skip_layer.append('fc6')
      fc6 = fc(flattened, flattened_shape, self.fc, name='fc6')
      dropout6 = dropout(fc6, self.keep_prob)

      fc7 = fc(dropout6, self.fc, self.fc, name='fc7')
      dropout7 = dropout(fc7, self.keep_prob)

      self.fc8 = fc(dropout7, self.fc, self.num_classes, relu=False, name='fc8')

  def load_initial_weights(self, session):
      weights_dict = np.load(self.weights_path, encoding='bytes').item()

      for op_name in weights_dict:
          if op_name not in self.skip_layer:
              with tf.variable_scope(op_name, reuse=True):
                  for data in weights_dict[op_name]:
                      if len(data.shape) == 1:

                          var = tf.get_variable('biases', trainable=False)
                          session.run(var.assign(data))
                      else:

                          var = tf.get_variable('weights', trainable=False)
                          session.run(var.assign(data))


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding="SAME", groups=1):

    input_channels = int(x.get_shape()[-1])


    # lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layers
        weights =  tf.get_variable('weights',
            shape=[filter_height, filter_width, input_channels / groups,
                   num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

        if groups == 1:
            conv = convolve(x, weights)

        else:

            input_groups = tf.split(num_or_size_splits=groups, axis=3, value=x)
            weight_groups = tf.split(num_or_size_splits=groups, axis=3,
                                     value=weights)
            output_groups = [convolve(i, k) for i, k in
                             zip(input_groups, weight_groups)]

            conv = tf.concat(axis=3, values=output_groups)

        bias = tf.reshape(tf.nn.bias_add(conv, biases),
                          conv.get_shape().as_list())

        relu = tf.nn.relu(bias, name=scope.name)

        return relu


def fc(x, num_in, num_out, name, relu=True):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
        biases = tf.get_variable('biases', shape=[num_out], trainable=True)

        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu == True:
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.lrn(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias,
                     name=name)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)
