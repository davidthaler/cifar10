'''
This script runs the keras ImageDataGenerator on some of the
Cifar10/Kaggle input .png files.

NB: all unknown arguments are passed to the constructor of
    keras.preprocessing.image.ImageDataGenerator
'''
from argparse import ArgumentParser
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 20

def parse_unknown_args(arglist):
    '''
    Takes the list of unknown arguments from ArgumentParser.parse_known_args
    and turns it into a dict that can be used as **args for the 
    ImageDataGenerator, which has a lot of parameters.

    Args:
        arglist: the list returned as the second value from
            ArgumentParser.parse_known_args
    
    Returns:
        dict with even-indexed elements of arglist as keys and odd-index
        elements as values; basic conversion is performed on the values
    '''
    keys = [k.strip('-') for k in arglist[::2]]
    values = []
    for val in arglist[1::2]:
        if val == 'True':
            values.append(True)
        elif val == 'False':
            values.append(False)
        elif val.isnumeric():
            values.append(int(val))
        else:
            try:
                values.append(float(val))
            except ValueError:
                values.append(val)
    return dict(zip(keys, values))

def run(inpath, outpath, size, batches, datagen_args):
    '''
    Makes an ImageDataGenerator using datagen_args, and runs it on
    images from inpath, putting results at outpath.
    '''
    datagen = ImageDataGenerator(**datagen_args)
    images = datagen.flow_from_directory(str(inpath),
                                         save_to_dir=str(outpath),
                                         target_size=(size, size),
                                         class_mode=None,
                                         interpolation='lanczos',
                                         batch_size=BATCH_SIZE)
    for k in range(batches):
        next(images)

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Unknown arguments passed to keras ImageDataGenerator')
    parser.add_argument('--inpath', required=True,
        help='Path to input directory')
    parser.add_argument('--outpath', required=True,
        help='Path to output directory')
    parser.add_argument('--size', type=int, default=64,
        help='output images will be size x size')
    parser.add_argument('--number', type=int, default=100,
        help='number of images to process')
    args, unknown = parser.parse_known_args()
    datagen_args = parse_unknown_args(unknown)
    inpath = Path() / args.inpath
    outpath = Path() / args.outpath
    if not outpath.exists():
        outpath.mkdir(parents=True)
    run(inpath, outpath, args.size, args.number // BATCH_SIZE, datagen_args)