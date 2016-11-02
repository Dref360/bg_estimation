# Here, we write the code to train the model, we overfit the model on a video and evaluate the result
import argparse
import json
import logging

import numpy as np

from analyze.Evaluate import Evaluate
from data.database import Database
from lib.decorator import GeneratorLoop
from lib.utils import chunks, CSVLogger
from src.CRNN import CRNN
from src.c3d import C3DModel
from src.unet import UNETModel
from src.vae import VAE
from src.vgg3d import VGG3DModel

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--db_path", dest="db_path", default="../dataset", type=str, help="dataset path")
parser.add_argument("--weight_file", dest="weight_file", type=str, help="model weight to be loaded, blank if new model")
parser.add_argument("--sequence_size", dest="sequence_size", default=10, type=int, help="batch size")
parser.add_argument("--batch_size", dest="batch_size", default=1, type=int, help="batch size")
parser.add_argument("--n_epochs", dest="n_epochs", default=10, type=int, help="nb epochs")
parser.add_argument("--method", dest="method", default="c3d", type=str, help="[c3d,vgg,crnn,vae,unet]")
parser.add_argument("--ratio", dest="ratio", default=1.0, type=float, help="Ratio to separate train and test set")
batch_size = 1
options = parser.parse_args()

print(vars(options))
logging.basicConfig(filename='logging.log', level=logging.DEBUG,
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

db = Database(options.db_path, options.sequence_size, batch_size=options.batch_size, size=model.img_size,
              output_size=model.output_size)

n_epoch = 0
max_epoch = options.n_epochs


@GeneratorLoop
def get_generator_batched_for_id(id, ratio):
    for batch in chunks(db.get_datas_on_one(id)[:int(db.get_count_on_video(id) * ratio)], options.batch_size):
        imgs, gts = zip(*batch)
        yield model.preprocess(np.asarray([db.load_imgs(img) for img in imgs]),
                               np.asarray([db.get_groundtruth(gt, 255.0) for gt in gts]))


@GeneratorLoop
def get_generator_test_batched_for_id(id, ratio):
    for batch in chunks(db.get_datas_on_one(id)[int(db.get_count_on_video(id) * ratio):], options.batch_size):
        imgs, gts = zip(*batch)
        yield model.preprocess(np.asarray([db.load_imgs(img) for img in imgs]),
                               np.asarray([db.get_groundtruth(gt, 255.0) for gt in gts]))


head = ['AGE', 'pEPs', 'pCEPs', 'MSSSIM', 'PSNR', 'CQM']
report = {"report": {}}
for id in range(db.max_video):
    print("VIDEO : {}".format(id))
    model.get_model().fit_generator(generator=get_generator_batched_for_id(id, options.ratio),
                                    samples_per_epoch=int(db.get_count_on_video(id) * options.ratio),
                                    nb_epoch=max_epoch,
                                    callbacks=[CSVLogger("log.csv", append=True)])
    if db.get_count_on_video(id) * (1.0 - options.ratio) > 0:
        outputs = model.get_model().predict_generator(get_generator_test_batched_for_id(id,options.ratio), int(
            db.get_count_on_video(id) * (1.0 - options.ratio)))
        gt = db.get_groundtruth_from_id(id)
        gt = gt.reshape(list(gt.shape) + [1])
        acc = []
        for i, output in enumerate(outputs):
            acc.append(list(zip(head, Evaluate(gt, output))))
        report["report"]["{}_{}".format(db.videos[id]["input"], id)] = acc
json.dump(report, open("report{}.json".format(options.method), "w"))
