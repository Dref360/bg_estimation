# Here, we write the code to train the model
import argparse
import json

import cv2
import keras
import numpy as np

from data.database import Database
from lib.utils import chunks, CSVLogger
from lib.img_sim import compute_ssim
from src.c3d import C3DModel

parser = argparse.ArgumentParser()
parser.add_argument("--db_path", dest="db_path", default="../dataset", type=str, help="dataset path")
parser.add_argument("--model", dest="model", type=str, help="model weight to be loaded, blank if new model")
parser.add_argument("--sequence_size", dest="sequence_size", default=10, type=int, help="batch size")
parser.add_argument("--batch_size", dest="batch_size", default=1, type=int, help="batch size")
parser.add_argument("--custom_lenght", dest="custom_lenght", default=None, type=int, help="max video to look at")
parser.add_argument("--n_epochs", dest="n_epochs", default=10, type=int, help="nb epochs")
batch_size = 1
options = parser.parse_args()

model = C3DModel(options.sequence_size, batch_size=options.batch_size)
db = Database(options.db_path, options.sequence_size, batch_size=options.batch_size, size=model.img_size,
              output_size=model.output_size, custom_lenght=options.custom_lenght)

n_epoch = 0
max_epoch = options.n_epochs


def get_generator():
    for (imgs, gt) in db.get_datas():
        yield model.preprocess(np.asarray([db.load_imgs(imgs)]), gt)


def get_generator_batched():
    for batch in chunks(db.get_datas(), options.batch_size):
        imgs, gts = zip(*batch)
        yield model.preprocess(np.asarray([db.load_imgs(img) for img in imgs]), np.asarray(gts))


def get_validation_generator_batched():
    for batch in chunks(db.get_tests(), options.batch_size):
        imgs, gts = zip(*batch)
        yield model.preprocess(np.asarray([db.load_imgs(img) for img in imgs]), np.asarray(gts))


def save_one():
    imgs, gts = next(get_generator())

    img = model.get_model().predict(imgs)[0]
    output = 255.0 * np.reshape(img, [model.output_size, model.output_size])
    gt = 255. * np.reshape(gts, [model.output_size, model.output_size])
    cv2.imwrite("output.png", output)
    cv2.imwrite("gt.png", gt)
    return abs(compute_ssim(output,gt,255))


model.get_model().fit_generator(generator=get_generator_batched(), samples_per_epoch=db.get_total_count(),
                                nb_epoch=max_epoch,
                                callbacks=[keras.callbacks.ModelCheckpoint("mod.model"),
                                           CSVLogger("log.csv", append=True)])

history = model.get_model().evaluate_generator(get_validation_generator_batched(), db.get_total_test_count())
with open("history.log", "w") as f:
    json.dump(history, f)

print(save_one())

model.get_model().save_weights("{}_w.h5".format(model.name))
