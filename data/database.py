import os

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
        self.next_video()

    def get_groundtruth(self, ratio=1.0):
        return (cv2.resize(cv2.imread(self.videos[self.video_id]["gt"], 0),
                           (self.output_size, self.output_size))) / ratio

    def get_groundtruth_with_batch(self, ratio=1.0):
        gt = self.get_groundtruth(ratio)
        return np.asarray([np.copy(gt) for i in range(self.batch_size)])

    def next_video(self):
        self.video_id = (self.video_id + 1) % self.max_video
        self.id = 0
        self.base_dir = self.videos[self.video_id]["input"]
        self.current_inputs = os.listdir(self.videos[self.video_id]["input"])
        # Sort input so that there are in the correct order
        self.current_inputs = sorted(self.current_inputs, key=lambda x: int(x[2:-4]))
        return self.video_id == 0

    def get_total_count(self):
        return sum([len(os.listdir(self.videos[i]["input"])) for i in range(len(self.videos))])

    def next_batch(self):
        batch = []
        for _ in range(self.batch_size):
            batch1 = self.current_inputs[self.id:self.id + self.sequence_size]
            self.id += 1
            batch1 = [cv2.resize(cv2.imread(pjoin(self.base_dir, i)), (self.size, self.size)) for i in batch1]
            batch.append(batch1)
        return np.asarray(batch)


if __name__ == "__main__":
    p = "/home/fred/DeepLearning/dataset/"
    db = Database(p, 100)
    for i in range(100):
        x = db.next_batch()
        if None in x:
            break

    db.next_video()
    for i in range(100):
        x = db.next_batch()
        if None in x:
            break
    print(x)
