import os

import cv2
import numpy as np

pjoin = os.path.join


class Database:
    def __init__(self, path, sequence_size,batch_size=1, size=321):
        self.videos = []
        self.id = 0
        self.size = size
        self.video_id = -1
        self.batch_size=batch_size
        self.current_inputs = None
        self.sequence_size = sequence_size
        assert os.path.exists(path)
        for category in os.listdir(path):
            for video in os.listdir(pjoin(path, category)):
                v = {
                    "gt": pjoin(path, category, video) + "/GT_background1.jpg",
                    "input": pjoin(path, category, video) + "/input"
                }
                self.videos.append(v)
        self.next_video()

    def get_groundtruth(self):
        return cv2.resize(cv2.imread(self.videos[self.video_id]["gt"], 0), (self.size, self.size))

    def get_groundtruth_with_batch(self):
        gt = cv2.resize(cv2.imread(self.videos[self.video_id]["gt"], 0), (self.size, self.size))
        return np.asarray([np.copy(gt) for i in range(self.batch_size)])

    def next_video(self):
        self.video_id += 1 % len(self.videos)
        self.id = 0
        self.base_dir = self.videos[self.video_id]["input"]
        self.current_inputs = os.listdir(self.videos[self.video_id]["input"])
        # Sort input so that there are in the correct order
        self.current_inputs = sorted(self.current_inputs, key=lambda x: int(x[2:-4]))
        return self.video_id == 0

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
