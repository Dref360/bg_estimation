# Here, we write the code to train the model
import argparse
from src.c3d import C3DModel
from data.database import Database

parser = argparse.ArgumentParser()
parser.add_argument("--db_path", dest="db_path", default="../dataset", type=str, help="dataset path")
parser.add_argument("--model", dest="model", type=str, help="model weight to be loaded, blank if new model")
parser.add_argument("--sequence_size", dest="sequence_size", default=10, type=int, help="batch size")

options = parser.parse_args()
db = Database(options.db_path, options.sequence_size)

model = C3DModel(options.sequence_size)

gt = db.get_groundtruth()

batch = db.next_batch()
n_epoch = 0
max_epoch = 10
while n_epoch != max_epoch:
    model.train_on(batch, gt)
    if None in batch:
        if db.next_video():
            n_epoch += 1
            gt = db.get_groundtruth()
    batch = db.next_batch()

model.get_model().save_weight("{}_w.h5".format(model.name))
