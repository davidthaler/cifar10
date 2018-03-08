'''
Cifar10, in tensorflow, using the Estimator API.
In this (first) version, we load the data as numpy arrays.
Keras is used in that.
'''
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf

BASE = Path.home() / 'Documents' / 'tensorflow' / 'cifar10'
DATA = BASE / 'data'
TRAIN = DATA / 'train'


def get_all_train():
    '''
    Get a 2-tuple of lists (filenames, labels) for the whole training set.
    '''
    labels = pd.read_csv(DATA / 'full_labels.csv')
    filenames = ['%d.png' % id for id in labels.id]
    return filenames, list(labels.y)


def get_validation_split():
    '''
    Get 2-tuple of 2-tuples of lists (filenames, labels) for the training
    and validation sets of a 1/5 validation split.
    '''
    f, y = get_all_train()
    n = len(y)
    ftr = [f[k] for k in range(n) if k % 5 != 0]
    ytr = [y[k] for k in range(n) if k % 5 != 0]
    fte = [f[k] for k in range(n) if k % 5 == 0]
    yte = [y[k] for k in range(n) if k % 5 == 0]
    return (ftr, ytr), (fte, yte)


def _map_img(filename, label):
    '''
    Applied as a dataset.map() function to the result of get_dataset.
    This function actually loads the image data.

    Args:
        filename: one filename
        label: one (int) label

    Returns:
        2-tuple of decoded image and label
    '''
    img_string = tf.read_file(filename)
    img_decode = tf.image.decode_image(img_string)          # This is unit8
    img_decode = tf.cast(img_decode, tf.float32) / 255.0
    img_decode.set_shape((32, 32, 3))
    img_decode = tf.image.per_image_standardization(img_decode)
    return img_decode, label


def cifar_dataset(filenames, labels, data_dir='train'):
    '''
    Get a tf.data.dataset of images from DATA/train (default) or DATA/test.

    Args:
        filenames: list of filenames within DATA / data_dir
        labels: list of (int) labels
        data_dir: (train|test) directory to get images from default train
    
    Returns:
        tf.data.dataset of paths and labels
    '''
    filenames = [str(DATA / data_dir / f) for f in filenames]
    filenames = tf.constant(filenames)
    # to allow handling test:
    # if labels is not None:
    labels = tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    return dataset  


def test_iterator(dataset, batchsize):
    '''
    Make a one-pass, in-order, batched-data iterator for test/validation
    from a dataset of filenames and labels.
    '''
    data_img = dataset.map(_map_img)
    data_batch = data_img.batch(batchsize)
    return data_batch.make_one_shot_iterator()


def train_iterator(dataset, batchsize):
    '''
    Make a data-shuffling, repeating, batched-data iterator for training
    from a dataset of filenames and labels.
    '''
    data_img = dataset.map(_map_img)
    data_batch = data_img.shuffle(1000).repeat().batch(batchsize)
    return data_batch.make_one_shot_iterator()


def get_tr_input_fn(batchsize=100):
    # NB: these need to return a function with signature:
    #  ()->(dict of `features`, `targets`)
    # ...it is not obvious that this has it...
    # but it looks like features and labels are graph ops
    (ftr, ytr), dummy = get_validation_split()
    train_dataset = cifar_dataset(ftr, ytr)
    tr_iter = train_iterator(train_dataset, batchsize=batchsize)
    features, labels = tr_iter.get_next()
    return {'x': features}, labels


def get_eval_input_fn(batchsize=100):
    dummy, (fval, yval) = get_validation_split()
    val_dataset = cifar_dataset(fval, yval)
    val_iter = test_iterator(val_dataset, batchsize=batchsize)
    features, labels = val_iter.get_next()
    return {'x':features}, labels
