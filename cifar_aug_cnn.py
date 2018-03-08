# Second attempt at Cifar10 data. This time with data augmentation.
# Gets 80 at 45-50 epochs when called with:
# >> python cifar_aug_cnn.py --width_shift 0.1 --height_shift 0.1 
# --dropout 0.25 --epochs 50 --rotate 5 --zoom 0.1
#
# Date 06-Feb-2018
import os
import argparse

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from cifar_util import get_cifar, make_validation_split, load_train

from keras.layers import BatchNormalization, Activation, Cropping2D

def get_data(args):
    '''
    Either use the Kaggle Cifar10 data with or without a validation set, or
    the data built in to keras and (ab)use its test set as validation data.
    '''
    if args.datasource == 'keras':
        xtr, ytr, xval, yval =  get_cifar()
        val = (xval, yval)
    elif args.datasource == 'validate':
        xtr, ytr, xval, yval =  make_validation_split()
        val = (xval, yval)
    else:                       # args was "full"
        xtr, ytr = load_train()
        val = None
    return xtr, ytr, val

def get_datagen(args):
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=args.width_shift,
                                 height_shift_range=args.height_shift,
                                 zoom_range=args.zoom,
                                 rotation_range=args.rotate,
                                 samplewise_center=args.center,
                                 samplewise_std_normalization=args.scale)
    return datagen

def get_valgen(args):
    valgen = ImageDataGenerator(samplewise_center=args.center,
                                samplewise_std_normalization=args.scale)
    return valgen

def build_model(args):
    model = Sequential()
    model.add(Cropping2D(2, input_shape=(32, 32, 3)))
    model.add(Conv2D(filters=args.filters1,
                     kernel_size=5))
    if args.batchnorm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=args.filters2,
                     kernel_size=3,
                     padding='same'))
    if args.batchnorm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=args.filters3,
                     kernel_size=3,
                     padding='same'))
    if args.batchnorm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(args.dense))
    if args.batchnorm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate=args.dropout))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def run(args):
    savepath = os.path.join(args.base_dir, args.name)
    xtr, ytr, val = get_data(args)
    datagen = get_datagen(args)
    if val is not None:
        if args.center or args.scale:
            valgen = get_valgen(args)
            xval, yval = val
            val = valgen.flow(xval, yval,
                              batch_size=args.batch_size,
                              shuffle=False)
    if args.restore:
        model = load_model(savepath)
    else:
        model = build_model(args)
    callbacks = None
    if args.save:
        if not os.path.exists(args.base_dir):
            os.makedirs(args.base_dir)
        callbacks = [ModelCheckpoint(savepath, verbose=1)]
    model.fit_generator(datagen.flow(xtr, ytr,
                        batch_size=args.batch_size),
                        validation_data=val,
                        workers=4,
                        epochs=args.epochs,
                        callbacks=callbacks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--center', action='store_true',
        help='set mean of each sample to 0')
    parser.add_argument('--scale', action='store_true',
        help='standard scale each image')
    parser.add_argument('--zoom', type=float, default=0.0,
        help='float zoom range; scaled to 1 +- zoom; default 0.0')
    parser.add_argument('--rotate', type=int, default=0,
        help='int rotation range in degrees; default 0')
    parser.add_argument('--height_shift', type=float, default=0.0,
        help='range for random vertical shifts in data augmentation;'+
             'default 0.0 for no shift')
    parser.add_argument('--width_shift', type=float, default=0.0,
        help='range for random horizontal shifts in data augmentation;'+
             'default 0.0 for no shift')
    parser.add_argument('--batch_size', type=int, default=100,
        help='training batch size; default 100')
    parser.add_argument('--epochs', type=int, default=1,
        help='number of training epochs; default 1')
    parser.add_argument('--dropout', type=float, default=0.25, 
        help='dropout rate (not retention); 0.0 is no dropout; default 0.25')
    parser.add_argument('--filters1', type=int, default=32,
        help='Number of filters in first convolutional layer; default 32')
    parser.add_argument('--filters2', type=int, default=64,
        help='Number of filters in second convolutional layer; default 64')
    parser.add_argument('--filters3', type=int, default=128,
        help='Number of filters in second convolutional layer; default 128')
    parser.add_argument('--dense', type=int, default=256,
        help='Number of units in dense, fully-connected layer; default 256')
    parser.add_argument('--batchnorm', action='store_true',
        help='Apply batchnorm on all layers')
    parser.add_argument('--name', default='',
        help='model directory is <base_dir>/<name>; default '' for <base_dir>')
    parser.add_argument('--base_dir', default='../models',
        help='base of estimator model_dir; default /tmp/cifar10aug/')
    parser.add_argument('--restore', action='store_true',
        help='restore from base_dir/name; if set, will ignore filtersN/dense')
    parser.add_argument('--save', action='store_true',
        help='save model at base_dir/name; overwrites anything there')
    parser.add_argument('--datasource', required=True,
        choices=['keras', 'full', 'validate'],
        help='Use the Kaggle data, optionally with a 1/5 validation set \n' + 
        'or use the data built in to keras.datasets')
    args, _ = parser.parse_known_args()
    run(args)
