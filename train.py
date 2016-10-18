# Here, we write the code to train the model
import argparse
import json
import logging

import cv2
import keras
import numpy as np

from data.database import Database
from lib.img_sim import compute_ssim
from lib.utils import chunks, CSVLogger
from src.CRNN import CRNN
from src.c3d import C3DModel
from src.vae import VAE
from src.vgg3d import VGG3DModel

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--db_path", dest="db_path", default="../dataset", type=str, help="dataset path")
parser.add_argument("--weight_file", dest="weight_file", type=str, help="model weight to be loaded, blank if new model")
parser.add_argument("--sequence_size", dest="sequence_size", default=10, type=int, help="batch size")
parser.add_argument("--batch_size", dest="batch_size", default=1, type=int, help="batch size")
parser.add_argument("--custom_lenght", dest="custom_lenght", default=None, type=int, help="max video to look at")
parser.add_argument("--n_epochs", dest="n_epochs", default=10, type=int, help="nb epochs")
parser.add_argument("--method", dest="method", default="c3d", type=str, help="[c3d,vgg,crnn,vae,gan]")
batch_size = 1
options = parser.parse_args()

print(vars(options))
logging.basicConfig(filename='logging.log', level=logging.DEBUG,
                    format='%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s')
logging.info(vars(options))

methods = ["c3d", "crnn", "vae", "gan", "vgg"]
assert options.method in methods, "Not a valid method"

if options.method == "c3d":
    model = C3DModel(options.sequence_size, batch_size=options.batch_size, weight_file=options.weight_file)
elif options.method == "crnn":
    model = CRNN(options.sequence_size, batch_size=options.batch_size, weight_file=options.weight_file)
elif options.method == "vae":
    model = VAE(options.sequence_size, batch_size=options.batch_size, weight_file=options.weight_file)
elif options.method == "vgg":
    model = VGG3DModel(options.sequence_size, batch_size=options.batch_size, weight_file=options.weight_file)
else:
    print("{} is not available at this moment".format(options.method))
    exit(0)

db = Database(options.db_path, options.sequence_size, batch_size=options.batch_size, size=model.img_size,
              output_size=model.output_size, custom_lenght=options.custom_lenght)

n_epoch = 0
max_epoch = options.n_epochs


def get_generator():
    while True:
        for (imgs, gt) in db.get_datas():
            yield model.preprocess(np.asarray([db.load_imgs(imgs)]), db.get_groundtruth(gt, 255.0))


def get_generator_batched():
    while True:
        for batch in chunks(db.get_datas(), options.batch_size):
            imgs, gts = zip(*batch)
            yield model.preprocess(np.asarray([db.load_imgs(img) for img in imgs]),
                                   np.asarray([db.get_groundtruth(gt, 255.0) for gt in gts]))


def get_validation_generator_batched():
    while True:
        for batch in chunks(db.get_tests(), options.batch_size):
            imgs, gts = zip(*batch)
            yield model.preprocess(np.asarray([db.load_imgs(img) for img in imgs]),
                                   np.asarray([db.get_groundtruth(gt, 255.0) for gt in gts]))


def save_one():
    imgs, gts = next(get_generator())

    img = model.get_model().predict(imgs)[0]
    output = 255.0 * np.reshape(img, [model.output_size, model.output_size])
    gt = 255. * np.reshape(gts, [model.output_size, model.output_size])
    cv2.imwrite("output.png", output)
    cv2.imwrite("gt.png", gt)
    return abs(compute_ssim(output, gt, 255))


try:
    model.get_model().fit_generator(generator=get_generator_batched(), samples_per_epoch=db.get_total_count(),
                                    nb_epoch=max_epoch,
                                    callbacks=[keras.callbacks.ModelCheckpoint("mod.model"),
                                               CSVLogger("log.csv", append=True)])
except Exception:
    logging.warning("Model stopped training!")

logging.info("Starting Testing")
history = model.get_model().evaluate_generator(get_validation_generator_batched(), db.get_total_test_count())
logging.info(history)
with open("history.log", "w") as f:
    json.dump(history, f)

ssms = save_one()
logging.info("SSMS {}".format(ssms))

model.get_model().save_weights("{}_w.h5".format(model.name))
