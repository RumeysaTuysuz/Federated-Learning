import tensorflow as tf
import numpy as np
from tensorflow.compat.v1.train import AdamOptimizer


def AlexNet(input_shape, num_classes, learning_rate, graph):
    """
        Construct the AlexNet model.
        input_shape: The shape of input (`list` like
        num_classes: The number of output classes (`int`)
        learning_rate: learning rate for optimizer (`float`)
        graph: The tf computation graph (`tf.Graph`)
    """
    with graph.as_default():
        X = tf.placeholder(tf.float32, input_shape, name='X')
        Y = tf.placeholder(tf.float32, [None, num_classes], name='Y')
        DROP_RATE = tf.placeholder(tf.float32, name='drop_rate')

        conv1 = conv(X, 11, 11, 96, 2, 2, name='conv1')
        norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')


        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')


        conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

     
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')


        flattened = tf.reshape(pool5, [-1, 1 * 1 * 256])
        fc6 = fc_layer(flattened, 1 * 1 * 256, 1024, name='fc6')
        dropout6 = dropout(fc6, DROP_RATE)

        fc7 = fc_layer(dropout6, 1024, 2048, name='fc7')
        dropout7 = dropout(fc7, DROP_RATE)


        logits = fc_layer(dropout7, 2048, num_classes, relu=False, name='fc8')

        loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                        labels=Y))
        optimizer = AdamOptimizer(
            learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

       
        prediction = tf.nn.softmax(logits)
        pred = tf.argmax(prediction, 1)

      
        correct_pred = tf.equal(pred, tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(
            tf.cast(correct_pred, tf.float32))

        return X, Y, DROP_RATE, train_op, loss_op, accuracy


def conv(x, filter_height, filter_width, num_filters,
            stride_y, stride_x, name, padding='SAME', groups=1):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
  
    input_channels = int(x.get_shape()[-1])


    convolve = lambda i, k: tf.nn.conv2d(
        i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights',
                                    shape=[
                                        filter_height, filter_width,
                                        input_channels / groups, num_filters
                                    ])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    else:
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3,
                                    num_or_size_splits=groups,
                                    value=weights)
        output_groups = [
            convolve(i, k) for i, k in zip(input_groups, weight_groups)
        ]

        conv = tf.concat(axis=3, values=output_groups)

    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    relu = tf.nn.relu(bias, name=scope.name)

    return relu


def fc_layer(x, input_size, output_size, name, relu=True, k=20):
    """Create a fully connected layer."""

    with tf.variable_scope(name) as scope:.
        W = tf.get_variable('weights', shape=[input_size, output_size])
        b = tf.get_variable('biases', shape=[output_size])
        z = tf.nn.bias_add(tf.matmul(x, W), b, name=scope.name)

    if relu:
        a = tf.nn.relu(z)
        return a

    else:
        return z


def max_pool(x,
                filter_height, filter_width,
                stride_y, stride_x,
                name, padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool2d(x,
        ksize=[1, filter_height, filter_width, 1],
        strides=[1, stride_y, stride_x, 1],
        padding=padding,
        name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x,
        depth_radius=radius,
        alpha=alpha,
        beta=beta,
        bias=bias,
        name=name)


def dropout(x, rate):
    """Create a dropout layer."""
    return tf.nn.dropout(x, rate=rate)
