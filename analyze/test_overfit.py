# Here, we write the code to train the model
import argparse
import logging

import cv2
import numpy as np

from analyze.Evaluate import Evaluate
from data.database import Database
from lib.decorator import GeneratorLoop
from lib.utils import chunks
from src.CRNN import CRNN
from src.c3d import C3DModel
from src.unet import UNETModel
from src.vae import VAE
from src.vgg3d import VGG3DModel

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--weight_file", type=str, default=None, help="model weight to be loaded, blank if new model")
parser.add_argument("--db_path", dest="db_path", default="../../dataset", type=str, help="dataset path")
parser.add_argument("--sequence_size", dest="sequence_size", default=10, type=int, help="batch size")
parser.add_argument("--batch_size", dest="batch_size", default=1, type=int, help="batch size")
parser.add_argument("--videoid", dest="videoid", default=0, type=int, help="max video to look at")
parser.add_argument("--method", dest="method", default="c3d", type=str, help="[c3d,vgg,crnn,vae,unet]")
batch_size = 1
options = parser.parse_args()

print(vars(options))
logging.basicConfig(filename='logging_overfit.log', level=logging.DEBUG,
                    format='%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s')
logging.info(vars(options))

methods = ["c3d", "crnn", "vae", "unet", "vgg"]
assert options.method in methods, "Not a valid method"

if options.method == "c3d":
    model = C3DModel(options.sequence_size, batch_size=options.batch_size, weight_file=options.weight_file)
elif options.method == "crnn":
    model = CRNN(options.sequence_size, batch_size=options.batch_size, weight_file=options.weight_file)
elif options.method == "vae":
    model = VAE(options.sequence_size, batch_size=options.batch_size, weight_file=options.weight_file)
elif options.method == "vgg":
    model = VGG3DModel(options.sequence_size, batch_size=options.batch_size, weight_file=options.weight_file)
elif options.method == "unet":
    model = UNETModel(options.sequence_size, batch_size=options.batch_size, weight_file=options.weight_file)
else:
    print("{} is not available at this moment".format(options.method))
    exit(0)

output_file = "output/out{}_{}.png"

db = Database(options.db_path, options.sequence_size, batch_size=options.batch_size, size=model.img_size,
              output_size=model.output_size)
max_epoch = 15


@GeneratorLoop
def get_generator_batched_for_id(id):
    for batch in chunks(db.get_datas_on_one(id),
                        options.batch_size):
        imgs, gts = zip(*batch)
        yield model.preprocess(np.asarray([db.load_imgs(img) for img in imgs]),
                               np.asarray([db.get_groundtruth(gt, 255.0) for gt in gts]))


ratio = 0.5


@GeneratorLoop
def get_generator_test_batched_for_id(id):
    max_test = db.get_count_on_video(id) - options.max_length
    for batch in chunks(db.get_datas_on_one(id),
                        options.batch_size):
        imgs, gts = zip(*batch)
        yield model.preprocess(np.asarray([db.load_imgs(img) for img in imgs]),
                               np.asarray([db.get_groundtruth(gt, 255.0) for gt in gts]))


if options.weight_file is None:
    model.get_model().fit_generator(generator=get_generator_batched_for_id(options.videoid),
                                    samples_per_epoch=int(db.get_count_on_video(options.videoid) * ratio),
                                    nb_epoch=max_epoch)
outputs = model.get_model().predict_generator(get_generator_batched_for_id(options.videoid),
                                              int(db.get_count_on_video(options.videoid) * (1 - ratio)))
gt = db.get_groundtruth_from_id(options.videoid)
gt = gt.reshape(list(gt.shape) + [1])
acc = []
for i, output in enumerate(outputs):
    cv2.imwrite(output_file.format(options.method, i), output.reshape([model.output_size, model.output_size]) * 255.)
    acc += Evaluate(gt, output)

print(acc)
