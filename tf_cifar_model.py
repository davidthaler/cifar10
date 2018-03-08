# This code is an adaptation of an MNIST model to the Cifar10 data.
# 
# The MNIST model was a modified version of the TF layers tutorial at:
# https://www.tensorflow.org/tutorials/layers
#
# Date: 2018-03-06
import sys
import os
from pathlib import Path
import shutil
import argparse
from string import Template
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_cifar_data

RESULTS = Template('Iter: $global_step   Loss: $loss   Accuracy: $accuracy')

def cnn_model_fn(features, labels, mode, params):
    biasInit = tf.constant_initializer(0.1, tf.float32)
    conv1 = tf.layers.conv2d(
        inputs=features['x'],
        filters=params['filters1'],
        kernel_size=5,
        padding='valid',
        activation=tf.nn.relu,
        bias_initializer=biasInit)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, 
                                    pool_size=2,
                                    strides=2)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=params['filters2'],
        kernel_size=3,
        padding='same',
        activation=tf.nn.relu,
        bias_initializer=biasInit)
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=2,
                                    strides=2)
    pool2flat = tf.reshape(pool2, shape=[-1, 7 * 7 * params['filters2']])
    dense = tf.layers.dense(inputs=pool2flat,
                            units=params['dense'],
                            activation=tf.nn.relu,
                            bias_initializer=biasInit)
    dropout = tf.layers.dropout(inputs=dense,
                                rate=params['dropout'],
                                training=(mode==tf.estimator.ModeKeys.TRAIN))
    logits = tf.layers.dense(inputs=dropout, units=10)

    # Predictions used in PREDICT and EVAL modes
    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Loss used in EVAL and TRAIN modes
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    acc = tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
    tf.summary.scalar('accuracy', acc[1])
    tf.summary.histogram('logits', logits)
    tf.summary.histogram('pool1', pool1)
    tf.summary.histogram('pool2', pool2)

    # train op for TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learn_rate'])
        train_op = optimizer.minimize(loss=loss, 
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, 
                                          loss=loss, 
                                          train_op=train_op)

    # EVAL mode
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {'accuracy': acc}
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)



def main(args):

    params = {'filters1':args.filters1,
              'filters2':args.filters2,
              'dense':args.dense,
              'dropout': args.dropout,
              'learn_rate': args.learn_rate}

    cifar_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                              model_dir=args.model_dir,
                                              params=params)

    # The 40000 is only valid for a validation split
    batch_per_epoch = 40000 // args.batch_size
    tr_input_fn = lambda : tf_cifar_data.get_tr_input_fn(batchsize=args.batch_size)
    eval_input_fn = lambda : tf_cifar_data.get_eval_input_fn()

    for i in range(args.epochs):
        cifar_classifier.train(
            input_fn=tr_input_fn,
            steps=batch_per_epoch)
        eval_results = cifar_classifier.evaluate(input_fn=eval_input_fn)
        print(RESULTS.substitute(eval_results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run train/eval on cifar')
    parser.add_argument('--epochs', type=int, default=1,
        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=100,
        help='training batch size; default 100')
    parser.add_argument('--dropout', type=float, default=0.5, 
        help='dropout rate (not retention); 0.0 is no dropout; default 0.5')
    parser.add_argument('--learn_rate', type=float, default=0.001,
        help='Learning rate; default 0.001')
    parser.add_argument('--filters1', type=int, default=32,
        help='Number of filters in first convolutional layer; default 32')
    parser.add_argument('--filters2', type=int, default=64,
        help='Number of filters in second convolutional layer; default 64')
    parser.add_argument('--dense', type=int, default=128,
        help='Number of units in dense, fully-connected layer; default 128')
    parser.add_argument('--name', default='',
        help='model directory is <base_dir>/<name>; default '' for <base_dir>')
    parser.add_argument('--base_dir', default='/tmp/tf_cifar',
        help='base of estimator model_dir; default /tmp/tf_cifar')
    parser.add_argument('--overwrite', action='store_true',
        help='Overwrite any model at <base_dir>/<name>, ' 
            + 'otherwise continue training it, if present')
    parser.add_argument('--quiet', action='store_true', 
        help='emit minimal logging information')
    args, _ = parser.parse_known_args(sys.argv)
    args.model_dir = os.path.join(args.base_dir, args.name, '')
    if not args.quiet:
        tf.logging.set_verbosity(tf.logging.INFO)
    if os.path.exists(args.model_dir):
        if args.overwrite:
            shutil.rmtree(args.model_dir)
    main(args)
