# Here, we write the code to train the model
import argparse
import logging

import numpy as np

from analyze.Evaluate import Evaluate
from data.database import Database
from src.CRNN import CRNN
from src.c3d import C3DModel
from src.vae import VAE
from src.vgg3d import VGG3DModel
from lib.decorator import GeneratorLoop

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("weight_file", type=str, help="model weight to be loaded, blank if new model")
parser.add_argument("--db_path", dest="db_path", default="../../dataset", type=str, help="dataset path")
parser.add_argument("--sequence_size", dest="sequence_size", default=10, type=int, help="batch size")
parser.add_argument("--batch_size", dest="batch_size", default=1, type=int, help="batch size")
parser.add_argument("--videoid", dest="videoid", default=0, type=int, help="max video to look at")
parser.add_argument("--method", dest="method", default="c3d", type=str, help="[c3d,vgg,crnn,vae,gan]")
batch_size = 1
options = parser.parse_args()

print(vars(options))
logging.basicConfig(filename='logging_overfit.log', level=logging.DEBUG,
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
              output_size=model.output_size)


@GeneratorLoop
def overfit_generator():
    for (imgs, gt) in db.get_datas_on_one(options.videoid):
        yield model.preprocess(np.asarray([db.load_imgs(imgs)]), db.get_groundtruth(gt, 255.0))


outputs = model.get_model().predict_generator(overfit_generator(), db.get_count_on_video(options.videoid))
gt = db.get_groundtruth_from_id(options.videoid)
acc = []
for output in outputs:
    acc += Evaluate(gt, output)

print(acc)
