'''
Get the processed data from map_labels and process_cifar in to numpy arrays
that are ready to be fed into keras.

Date: 2018-02-26
'''
from pathlib import Path
import numpy as np
import pandas as pd 
from keras.utils import to_categorical
import pdb


BASE = Path.home() / 'Documents' / 'tensorflow' / 'cifar10'
DATA = BASE / 'data'
NPDATA = DATA / 'numpy_data'


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
        x = np.load(testpath / filename)
        yield x


def batch_predict(model):
    '''
    The data is borderline for fitting in memory,
    so we use large prediction batches.

    Args:
        model: a fitted model

    Returns:
        1-D vector of predicted class indices (integers)
    '''
    results = []
    for x in testgen():
        pred = model.predict(x).argmax(axis=1)
        results.append(pred)
    return np.concatenate(results)


def prepare_submission(preds):
    '''
    Load sample submission, converts prediction indices to labels,
    then writes predictions into sample submission frame.

    Args:
        preds: output from batch_predict
    '''
    # get pd.Series to map prediction indices to label names
    labelpath = DATA / 'full_labels.csv'
    fulllabels = pd.read_csv(labelpath)
    label_map = fulllabels.groupby('y').label.first()
    #pdb.set_trace()
    ss = pd.read_csv(DATA / 'sampleSubmission.csv')
    ss.label = preds
    ss.label = ss.label.map(label_map)
    return ss


def full_predict(model, submission_name):
    '''Predict on test set and write out submission.'''
    preds = batch_predict(model)
    sub = prepare_submission(preds)
    subname = BASE / 'submissions' / ('submission_%s.csv' % submission_name)
    sub.to_csv(subname, index=False)
