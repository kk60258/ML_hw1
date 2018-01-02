#!/usr/bin/env python
# -- coding: utf8 --

import tensorflow as tf

class HelloCNN:
    def __init__(self):
        self.x, self.output = self.__build_model(True)


    def __build_model(self, isTraining):
        #input shape = [size, 48, 48, 1]
        #input_layer = tf.reshape(input_img, [-1, 48, 48, 1])

        x = tf.placeholder(tf.float32, [None, 48, 48, 1])

        conv1 = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[5, 5], padding="valid", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[5, 5], strides=2, padding="same")
        out1 = pool1

        conv2 = tf.layers.conv2d(inputs=out1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        out2 = conv2

        conv3 = tf.layers.conv2d(inputs=out2, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        pool3 = tf.layers.average_pooling2d(inputs=conv3, pool_size=[3, 3], strides=2)
        out3 = pool3

        conv4 = tf.layers.conv2d(inputs=out3, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        out4 = conv4

        conv5 = tf.layers.conv2d(inputs=out4, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        pool5 = tf.layers.average_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)
        out5 = pool5

        # Dense Layer
        flat = tf.reshape(out5, [-1, 128 * 4 * 4])
        # flat = tf.layers.flatten(out5)
        dense6 = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
        dropout6 = tf.layers.dropout(inputs=dense6, rate=0.5, training=isTraining)

        dense7 = tf.layers.dense(inputs=dropout6, units=1024, activation=tf.nn.relu)
        dropout7 = tf.layers.dropout(inputs=dense7, rate=0.5, training=isTraining)

        # Logits Layer
        output = tf.layers.dense(inputs=dropout7, units=7, activation=tf.nn.softmax)

        return x, output