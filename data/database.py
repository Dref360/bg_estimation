import cv2
import numpy as np
import os
pjoin = os.path.join
class Database:
    def __init__(self,path,batch_size,size = 321):
        self.videos = []
        self.id = 0
        self.size = size
        self.video_id = -1
        self.current_inputs = None
        self.batch_size = batch_size
        assert os.path.exists(path)
        for category in os.listdir(path):
            for video in os.listdir(pjoin(path,category)):
                v = {
                    "gt":pjoin(path,category,video)+"/GT_background1.jpg",
                    "input":pjoin(path,category,video)+"/input"
                }
                self.videos.append(v)
        self.next_video()

    def next_video(self):
        self.video_id += 1
        self.id = 0
        self.base_dir = self.videos[self.video_id]["input"]
        self.current_inputs = os.listdir(self.videos[self.video_id]["input"])
        # Sort input so that there are in the correct order
        self.current_inputs = sorted(self.current_inputs,key=lambda x: int(x[2:-4]))

    def next_batch(self):
        batch = self.current_inputs[self.id:self.id+self.batch_size]
        self.id += self.batch_size
        batch = [cv2.resize(cv2.imread(pjoin(self.base_dir, i)),(self.size,self.size)) for i in batch]
        batch += [None] * (self.batch_size - len(batch))
        return batch

if __name__ == "__main__":
    p = "/home/fred/DeepLearning/dataset/"
    db = Database(p,100)
    for i in range(100):
        x = db.next_batch()
        if None in x :
            break

    db.next_video()
    for i in range(100):
        x = db.next_batch()
        if None in x :
            break
    print(x)


