# Here, we write the code to train the model
import argparse

import numpy as np

from data.database import Database
from src.c3d import C3DModel

parser = argparse.ArgumentParser()
parser.add_argument("--db_path", dest="db_path", default="../dataset", type=str, help="dataset path")
parser.add_argument("--model", dest="model", type=str, help="model weight to be loaded, blank if new model")
parser.add_argument("--sequence_size", dest="sequence_size", default=10, type=int, help="batch size")
batch_size = 1
options = parser.parse_args()
db = Database(options.db_path, options.sequence_size,batch_size=batch_size)

model = C3DModel(options.sequence_size)

n_epoch = 0
max_epoch = 10
for i in range(max_epoch):
    for vid in range(len(db.videos)):
        batch = db.next_batch()
        gt = db.get_groundtruth_with_batch()
        while len(batch) == batch_size:
            model.train_on(batch,gt)
            batch = db.next_batch()
        db.next_video()
    n_epoch += 1


model.get_model().save_weight("{}_w.h5".format(model.name))
