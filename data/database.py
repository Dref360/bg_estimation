import os
from random import shuffle

import cv2
import numpy as np

pjoin = os.path.join


class Database:
    def __init__(self, path, sequence_size, batch_size=1, size=321, output_size=None, custom_lenght=None):
        self.videos = []
        self.id = 0
        self.size = size
        self.video_id = -1
        self.batch_size = batch_size
        self.current_inputs = None
        self.sequence_size = sequence_size
        self.output_size = size if output_size is None else output_size
        assert os.path.exists(path)
        for category in os.listdir(path):
            for video in os.listdir(pjoin(path, category)):
                v = {
                    "gt": pjoin(path, category, video) + "/GT_background1.jpg",
                    "input": pjoin(path, category, video) + "/input"
                }
                self.videos.append(v)
        self.max_video = len(self.videos) if custom_lenght is None else custom_lenght

    def get_datas(self):
        acc = []
        for video in self.videos:
            gt = self.get_groundtruth(video["gt"], 255.0)
            current_inputs = os.listdir(self.videos[self.video_id]["input"])
            current_inputs = sorted(current_inputs, key=lambda x: int(x[2:-4]))
            current_inputs = [pjoin(self.videos[self.video_id]["input"], i) for i in current_inputs]
            acc += [(current_inputs[i:i + 10], gt) for i in range(len(current_inputs) - 10)]
        shuffle(acc)
        return acc

    def get_groundtruth(self, path, ratio=1.0):
        return (cv2.resize(cv2.imread(self.videos[self.video_id]["gt"], 0),
                           (self.output_size, self.output_size))) / ratio

    def get_groundtruth_with_batch(self, path, ratio=1.0):
        gt = self.get_groundtruth(ratio, path)
        return np.asarray([np.copy(gt) for i in range(self.batch_size)])

    def get_total_count(self):
        return sum([len(os.listdir(self.videos[i]["input"])) for i in range(self.max_video)])

    def load_imgs(self, imgs):
        return np.asarray([cv2.resize(cv2.imread(i), (self.size, self.size)) for i in imgs])