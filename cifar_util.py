'''
Load saved data, either from keras or Kaggle.
Make predictions/submission files.

Date: 2018-02-26
'''
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd 
from keras.utils import to_categorical
from keras.models import load_model
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from itertools import islice


BASE = Path.home() / 'Documents' / 'tensorflow' / 'cifar10'
DATA = BASE / 'data'
NPDATA = DATA / 'numpy_data'


def get_cifar():
    (xtr, ytr), (xte, yte) = cifar10.load_data()
    # y is a vector of labels in 0...9; change to one-hot
    ytr = to_categorical(ytr)
    yte = to_categorical(yte)
    # x is uint8 0...255; change to float in [0.0, 1.0]
    xtr = (xtr.astype('float32') / 255.0)
    xte = (xte.astype('float32') / 255.0)
    return xtr, ytr, xte, yte


def load_train():
    '''
    Loads the training data as prepared by map_labels and process_cifar.
    
    Returns:
        2-tuple x, y of training images and labels as numpy arrays.
        x is (50000, 32, 32, 3) and y is 50000 x 1
    '''
    labels = pd.read_csv(DATA / 'full_labels.csv')
    y = to_categorical(labels.y.values)
    datapath = NPDATA / 'train' / 'train.npy'
    x = np.load(datapath)
    return x, y


def make_validation_split():
    '''
    Make a 1/5 validation set, using load_train.

    Returns:
        4-tuple of xtrain, ytrain, xval, and yval
    '''
    x, y = load_train()
    idx = np.arange(len(y))
    tr_idx = idx[(idx % 5) != 0]
    val_idx = idx[(idx % 5) == 0]
    return (x[tr_idx], y[tr_idx], x[val_idx], y[val_idx])


def testgen():
    '''
    Generator for test batches of Kaggle Cifar10 data.

    Yields:
        6 batches of test data of size (50000, 32, 32, 3)
    '''
    testpath = NPDATA / 'test'
    for k in range(1, 7):
        filename = 'test_batch%d.npy' % k
        print('loading %s' % filename)
        x = np.load(testpath / filename)
        yield x


def scalegen(center, scale):
    '''
    Alternate test image data generator to use if data is centered/scaled.

    Args:
        center: samplewise center data
        scale: samplewise scale data

    Yields:
        batches of adjusted test images
    '''
    img_gen = ImageDataGenerator(samplewise_center=center,
                                 samplewise_std_normalization=scale)
    for x in testgen():
        yield from islice(img_gen.flow(x, batch_size=1000, shuffle=False), 50)


def batch_predict(model, data):
    '''
    The data is borderline for fitting in memory,
    so we use large prediction batches.

    Args:
        model: a fitted model
        data: generator for test data batches

    Returns:
        1-D vector of predicted class indices (integers)
    '''
    results = []
    for x in data:
        pred = model.predict(x).argmax(axis=1)
        results.append(pred)
    return np.concatenate(results)


def write_submission(predictions, submission_name):
    '''
    Writes out a submission file using predictions at
    submission_<submission_name>.csv

    Args:
        predictions: output from batch_predict
        submission_name: result named submission_<submission_name>.csv
    '''
    # get pd.Series to map prediction indices to label names
    labelpath = DATA / 'full_labels.csv'
    fulllabels = pd.read_csv(labelpath)
    label_map = fulllabels.groupby('y').label.first()
    ss = pd.read_csv(DATA / 'sampleSubmission.csv')
    ss.label = predictions
    ss.label = ss.label.map(label_map)
    subname = BASE / 'submissions' / ('submission_%s.csv.gz' % submission_name)
    ss.to_csv(subname, index=False, compression='gzip')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_dir', default=BASE / 'models',
        help='path to folder containing saved model')
    parser.add_argument('--sub', required=True,
        help='submission saved at: BASE/submissions/submission_<sub>.csv')
    parser.add_argument('--model_name', required=True,
        help='name of model in <model_dir>')
    parser.add_argument('--center', action='store_true',
        help='samplewise center the input')
    parser.add_argument('--scale', action='store_true',
        help='samplewise standard scale the input')
    args = parser.parse_args()
    modelpath = args.model_dir / args.model_name
    model = load_model(modelpath)
    if args.center or args.scale:
        datagen = scalegen(args.center, args.scale)
    else:
        datagen = testgen()
    preds = batch_predict(model, datagen)
    write_submission(preds, args.sub)
