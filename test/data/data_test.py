import unittest
from data.database import Database
import numpy as np
import os

class DatabaseTest(unittest.TestCase):
    def setUp(self):
        self.db = Database("/home/fred/DeepLearning/dataset/",10,1)

    def find_minimal_video(self):
        video_id = np.argmin(np.array([len(os.listdir(self.db.videos[i]["input"])) for i in range(len(self.db.videos))]))
        return video_id

    def test_database_const(self):
        self.db.video_id = self.find_minimal_video() - 1
        self.db.next_video()
        gt = self.db.get_groundtruth()
        batch = self.db.next_batch()
        for _ in range(10):
            if len(batch.shape) == 2:
                self.assertEqual((1,10,321,321,3),batch.shape)
            else:
                self.db.next_video()
                batch = self.db.next_batch()

            try:
                print(self.db.video_id)
                gt = self.db.get_groundtruth_with_batch()
                batch = np.transpose(batch, [0, 4, 1, 2, 3])
                batch = np.array(batch)
                gt = gt.reshape([self.db.batch_size, 1, self.db.size, self.db.size])
            except Exception:
                self.fail("Error!")
            batch = self.db.next_batch()

    def test_gt_loading(self):
        "assert that the DB is alright"
        for i in range(self.db.video_id):
            try:
                gt = self.db.get_groundtruth()
            except Exception:
                self.fail("NOT GOOD for {}".format(self.db.videos[i]["gt"]))
            self.db.next_video()
