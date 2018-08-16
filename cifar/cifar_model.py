import tensorflow as tf
from tensorflow.contrib import layers
import pdb
import numpy as np


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([
        x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


# initializer = tf.truncated_normal_initializer(stddev=0.02)
initializer = tf.contrib.layers.xavier_initializer()

Z_dim = 128 * 2
mb_size = 32 * 2
noise_dim = 20

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def discriminator(x, y):

    h = tf.reshape(x, [-1, 32, 32, 3])
    # noise = tf.random_normal([mb_size, 64, 64, 1])
    noise = tf.random_uniform([mb_size, 32, 32, 1], -1, 1)
    h = tf.concat([h, noise], axis=3)

    h = layers.conv2d(h, 64, 5, stride=2, padding='SAME',
                      activation_fn=None, weights_initializer=initializer)
    h = layers.batch_norm(h, activation_fn=lrelu)

    h = layers.conv2d(h, 64 * 2, 5, stride=2, padding='SAME',
                      activation_fn=None, weights_initializer=initializer)
    h = layers.batch_norm(h, activation_fn=lrelu)

    h = layers.conv2d(h, 64 * 4, 5, stride=2, padding='SAME',
                      activation_fn=None, weights_initializer=initializer)
    h = layers.batch_norm(h, activation_fn=lrelu)

    h = layers.conv2d(h, 64 * 8, 5, stride=2, padding='SAME',
                      activation_fn=None, weights_initializer=initializer)
    h = layers.batch_norm(h, activation_fn=lrelu)
    # h = layers.avg_pool2d(h, 2, stride=2)
    h = layers.flatten(h)
    noise_z = tf.random_uniform([mb_size, noise_dim], -1, 1)
    y = tf.concat([y, noise_z], axis=1)
    zh = layers.fully_connected(y, 2*2*512, activation_fn=lrelu)
    h = tf.concat([h, zh], axis=1)
    h = layers.fully_connected(h, 1, activation_fn=tf.identity)

    return h, tf.sigmoid(h)


def encoder1(tensor):
    # noise = tf.random_normal([mb_size, 64, 64, 1])
    noise = tf.random_uniform([mb_size, 32, 32, 1], -1, 1)
    tensor = tf.concat([tensor, noise], axis=3)
    conv1 = layers.conv2d(tensor, 32, 5, stride=2,
                          activation_fn=None, weights_initializer=initializer)
    conv1 = layers.batch_norm(conv1, activation_fn=tf.nn.relu)

    conv2 = layers.conv2d(conv1, 64, 5, stride=2, activation_fn=None,
                          normalizer_fn=layers.batch_norm, weights_initializer=initializer)
    conv2 = layers.batch_norm(conv2, activation_fn=tf.nn.relu)

    conv3 = layers.conv2d(conv2, 128, 5, stride=2, activation_fn=None, normalizer_fn=layers.batch_norm,
                          weights_initializer=initializer)
    conv3 = layers.batch_norm(conv3, activation_fn=tf.nn.relu)

    conv4 = layers.conv2d(conv3, 256, 5, stride=2, activation_fn=None, normalizer_fn=layers.batch_norm,
                          weights_initializer=initializer)
    conv4 = layers.batch_norm(conv4, activation_fn=tf.nn.relu)

    fc1 = layers.flatten(conv4)
    # fc1 = layers.fully_connected(
    #     inputs=fc1, num_outputs=512, activation_fn=None, weights_initializer=initializer)
    # fc1 = layers.batch_norm(fc1, activation_fn=tf.nn.relu)

    fc2 = layers.fully_connected(inputs=fc1, num_outputs=Z_dim,
                                 activation_fn=tf.nn.tanh, weights_initializer=initializer)

    return fc2


def encoder2(y):
    # noise = tf.random_normal([mb_size, 10])
    noise = tf.random_uniform([mb_size, noise_dim], -1, 1)
    h = tf.concat([y, noise], axis=1)

    # h = layers.fully_connected(h, 1024, weights_initializer=initializer)
    # h = layers.batch_norm(h, activation_fn=tf.nn.relu)

    h = layers.fully_connected(
        h, 64 * 8 * 4 * 4, activation_fn=None, weights_initializer=initializer)
    h = tf.reshape(h, [-1, 4, 4, 64 * 8])
    h = layers.batch_norm(h, activation_fn=tf.nn.relu)

    h = layers.conv2d_transpose(h, 64 * 4, 5, stride=2, padding='SAME',
                                activation_fn=None, weights_initializer=initializer)
    h = layers.batch_norm(h, activation_fn=lrelu)

    h = layers.conv2d_transpose(h, 64 * 2, 5, stride=2, padding='SAME',
                                activation_fn=None, weights_initializer=initializer)
    h = layers.batch_norm(h, activation_fn=tf.nn.relu)

    # h = layers.conv2d_transpose(h, 64 * 1, 5, stride=2, padding='SAME',
    #                             activation_fn=None, weights_initializer=initializer)
    # h = layers.batch_norm(h, activation_fn=None)

    h = layers.conv2d_transpose(h, 3, 5, stride=2, padding='SAME',
                                activation_fn=tf.nn.tanh, weights_initializer=initializer)
    return h
