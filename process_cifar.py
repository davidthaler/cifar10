'''
Process the Kaggle Cifar10 data to make it usable with the keras models.
The data and labels are different from what is provided in the Kaggle data.
NB: the labels are already processed using the map_labels.py script.

Date: 2018-02-26
'''
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from keras.preprocessing.image import img_to_array
import pdb

FLAGS = None
BASE = Path.home() / 'Documents' / 'tensorflow' / 'cifar10'
DATA = BASE / 'data'

def convert_batch(dir_name, start=1, stop=50000):
    '''
    Read in a lot of .png files like DATA/dir_name/<1...50000>.png
    Convert to ndarray with shape (1, 32, 32, 3)
    Append results to list; concatenate results and return

    Args:
        dir_name: ('test'|'train') load files from DATA/dir_name
        start: default 1, first file number to load
        stop: default 50000, last file number to load (eg inclusive)

    Returns:
        4-D ndarray of float32, shape (stop - start + 1, 32, 32, 3)
    '''
    loadpath = DATA / dir_name
    results = []
    for k in range(start, stop + 1):
        filename = '%d.png' % k
        img = Image.open(loadpath / filename)
        np_img = img_to_array(img)
        np_img = np_img.reshape((1, 32, 32, 3))
        np_img /= 255.0
        results.append(np_img)
    return np.concatenate(results)

def make_train():
    '''
    Make the training set as one array of shape (50000, 32, 32, 3)
    and save it.
    '''
    out_dir = DATA / FLAGS.outpath / 'train'
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    outname = out_dir / 'train.npy'
    data = convert_batch('train')
    np.save(outname, data)

def make_test():
    '''Make the test set as 6 arrays of shape (50000, 32, 32, 3)'''
    out_dir = DATA / FLAGS.outpath / 'test'
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    for k in range(1, 7):
        outname = out_dir / ('test_batch%d.npy' % k)
        end = 50000 * k
        start = end - 50000 + 1
        data = convert_batch('test', start, end)
        np.save(outname, data)
        print('Completed batch %d' % k)

if __name__ == '__main__':
    parser = ArgumentParser('Paths are relative to DATA, which is %s' % DATA)
    parser.add_argument('--outpath', default='numpy_data',
        help='output is written at DATA/outpath')
    parser.add_argument('--what', choices=['train', 'test'],
        help='make one of (train|test)')
    FLAGS, _  = parser.parse_known_args()
    #pdb.set_trace()
    if FLAGS.what == 'train':
        make_train()
    else:
        make_test()
