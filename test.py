# Here, we write the code to test the model
import argparse

import numpy as np

from data.database import Database
from src.c3d import C3DModel
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--db_path", dest="db_path", default="../dataset", type=str, help="dataset path")
parser.add_argument("--model", dest="model", type=str, help="model weight to be loaded, blank if new model")
parser.add_argument("--sequence_size", dest="sequence_size", default=10, type=int, help="batch size")

options = parser.parse_args()
db = Database(options.db_path, options.sequence_size)

model = C3DModel(options.sequence_size)

n_epoch = 0
max_epoch = 10
batch = db.next_batch()
while n_epoch != max_epoch:
    prediction = model.test_on(np.asarray(batch))
    for pred in prediction:
        cv2.imshow("pred",pred.reshape([321,321]))
    if np.any((batch == None)):
        print("NEW VIDEO")
        if db.next_video():
            n_epoch += 1
            print("NEW EPOCH")
    batch = db.next_batch()

model.get_model().save_weight("{}_w.h5".format(model.name))
