from pathlib import Path
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser(description="Convert Cifar data")
parser.add_argument("-srcdir", type=Path)
parser.add_argument("-outpkl", type=Path)
args = parser.parse_args()
cifar_top_level = args.srcdir
batch_names = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
    "test_batch"
]
di = {}
import pickle
for batch_name in batch_names:
    with open(cifar_top_level/batch_name, 'rb') as f:
        di[batch_name] = pickle.load(f, encoding='bytes')
    with open(cifar_top_level/"batches.meta", 'rb') as f:
        labels = pickle.load(f)
    label_idx_to_name = {i:l for i, l in enumerate(labels['label_names'])}

tot_dat = np.vstack([d[b'data'] for d in di.values()])
labels = np.concatenate([d[b'labels'] for d in di.values()])
dat = {"data": tot_dat,
       "labels": labels,
       "label_ids": label_idx_to_name}
import pickle
with open(args.outpkl, 'wb') as f:
    pickle.dump(dat, f)
