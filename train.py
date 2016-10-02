# Here, we write the code to train the model
import argparse

from data.database import Database
from src.c3d import C3DModel
import keras

parser = argparse.ArgumentParser()
parser.add_argument("--db_path", dest="db_path", default="../dataset", type=str, help="dataset path")
parser.add_argument("--model", dest="model", type=str, help="model weight to be loaded, blank if new model")
parser.add_argument("--sequence_size", dest="sequence_size", default=10, type=int, help="batch size")
batch_size = 1
options = parser.parse_args()
db = Database(options.db_path, options.sequence_size, batch_size=batch_size, size=321, output_size=158)

model = C3DModel(options.sequence_size, output_size=158)

n_epoch = 0
max_epoch = 10


def logging(epoch, log):
    with open("log.log", "w") as f:
        f.write(epoch)
        f.write(log)


def get_generator():
    for i in range(max_epoch):
        print("EPOCH {}".format(i))
        for vid in range(len(db.videos)):
            batch = db.next_batch()
            gt = db.get_groundtruth_with_batch(255.0)
            while len(batch) == batch_size and batch.shape[1] == options.sequence_size:
                yield model.preprocess(batch, gt)
                batch = db.next_batch()
            db.next_video()


model.get_model().fit_generator(get_generator(), db.get_total_count(), max_epoch,callbacks=[keras.callbacks.ModelCheckpoint("mod.model",save_best_only=True)])

model.get_model().save_weights("{}_w.h5".format(model.name))
