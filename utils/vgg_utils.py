import math
import tensorflow as tf
import numpy as np

def vgg_conv_layer(x, kernel_size, out_channels, stride, var_list, pad="SAME", name="conv"):
    """
    Define API for conv operation. This includes kernel declaration and
    conv operation both followed by relu.
    """
    in_channels = x.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        #n = kernel_size * kernel_size * out_channels
        n = kernel_size * in_channels
        stdv = 1.0 / math.sqrt(n)
        w = tf.get_variable('kernel_weights', [kernel_size, kernel_size, in_channels, out_channels],
                           tf.float32, 
                           initializer=tf.random_uniform_initializer(-stdv, stdv))
        b = tf.get_variable('kernel_biases', [out_channels], tf.float32, initializer=tf.random_uniform_initializer(-stdv, stdv))

        # Append the variable to the trainable variables list
        var_list.append(w)
        var_list.append(b)

    # Do the convolution operation
    bias = tf.nn.bias_add(tf.nn.conv2d(x, w, [1, stride, stride, 1], padding=pad), b)
    relu = tf.nn.relu(bias)
    return relu

def vgg_fc_layer(x, out_dim, var_list, apply_relu=True, name="fc"):
    """
    Define API for the fully connected layer. This includes both the variable
    declaration and matmul operation.
    """
    in_dim = x.get_shape().as_list()[1]
    stdv = 1.0 / math.sqrt(in_dim)
    with tf.variable_scope(name):
        # Define the weights and biases for this layer
        w = tf.get_variable('weights', [in_dim, out_dim], tf.float32, 
                initializer=tf.random_uniform_initializer(-stdv, stdv))
        b = tf.get_variable('biases', [out_dim], tf.float32, initializer=tf.random_uniform_initializer(-stdv, stdv))

        # Append the variable to the trainable variables list
        var_list.append(w)
        # var_list.append(b)

    # Do the FC operation
    x = tf.nn.l2_normalize(x, axis=1)  # theta 2020-09-01 ws
    w = tf.nn.l2_normalize(w, axis=0)  # theta 2020-09-01 ws
    # output = tf.matmul(x, w) + b
    output = tf.matmul(x, w)
    # Apply relu if needed
    if apply_relu:
        output = tf.nn.relu(output)

    return output
