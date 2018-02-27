# Script to add label numbers (used in keras) to the labels
# It looks like the label names are numbered in alphabetical order.
import pandas as pd
from pathlib import Path

BASE = Path.home() / 'Documents' / 'tensorflow' / 'cifar10'
LABELS = BASE / 'data' / 'trainLabels.csv'
labels = pd.read_csv(LABELS)
names = labels.label.unique()
names.sort()
label_map = dict(zip(names, range(10)))
labels['y'] = labels.label.map(label_map)
OUTPATH = BASE / 'data' / 'full_labels.csv'
labels.to_csv(OUTPATH, index=False)
