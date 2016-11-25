import os
from random import shuffle

import cv2
import numpy as np

pjoin = os.path.join


class Database:
    """Database object to handle the SBMnet dataset"""
    def __init__(self, path, sequence_size, batch_size=1, size=321, output_size=None, custom_lenght=None):
        """
        Initialize the Database, no loading at this point.
        :param path: Path to the database root
        :param sequence_size: Number of frame to return by call
        :param batch_size: batchsize to return
        :param size: Input image resized shape
        :param output_size: Groundtruth resized shape
        :param custom_lenght: Maximum of video to process (optional)
        """
        self.videos = []
        self.size = size
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
        shuffle(self.videos)
        self.max_video = len(self.videos) if custom_lenght is None else custom_lenght

    def get_datas(self):
        """
        Get all the datas shuffled up to {custom_lenght}.
        :return: [custom_lenght,N, sequence_size], where N is the number of sequence in the video, elements are path to actual images
        """
        acc = []
        for id in range(self.max_video):
            acc += self.get_datas_on_one(id)
        shuffle(acc)
        return acc

    def get_datas_on_one(self, id):
        """
        Get the data on one video
        :param id: video id
        :return: [N,sequence_size] where N is the number of sequence in this video.
        """
        current_inputs = os.listdir(self.videos[id]["input"])
        current_inputs = sorted(current_inputs, key=lambda x: int(x[2:-4]))
        current_inputs = [pjoin(self.videos[id]["input"], i) for i in current_inputs]
        res = [(current_inputs[i:i + self.sequence_size], self.videos[id]["gt"]) for i in
               range(len(current_inputs) - self.sequence_size)]
        return res

    def get_groundtruth_from_id(self, id):
        """
        Get the groundtruth for this video. normalized by 255.
        :param id: video id
        :return: The groundtruth image.
        """
        video = self.videos[id]
        gt = self.get_groundtruth(video["gt"], 255.0)
        return gt

    def get_tests(self):
        """
        Get the test set from {custom_lenght} to the end.
        :return: [M,N, sequence_size], where N is the number of sequence in the video, M is the difference between {custom_lenght} and the lenght of the dataset, elements are path to actual images
        """
        acc = []
        for id in range(self.max_video, len(self.videos)):
            acc += self.get_datas_on_one(id)
        shuffle(acc)
        return acc

    def get_groundtruth(self, path, ratio=1.0):
        """
        Load the image and apply a Rescale
        :param path: path to the image
        :param ratio: Maximum value
        :return: Image rescaled
        """
        return (cv2.resize(cv2.imread(path, 0),
                           (self.output_size, self.output_size))) / ratio

    def get_groundtruth_with_batch(self, path, ratio=1.0):
        """
        Get the groundtruth time the batch size.
        :param path: Path to the image
        :param ratio: maximum value
        :return: batchsize * [gt]
        """
        gt = self.get_groundtruth(path,ratio)
        return np.asarray([np.copy(gt) for i in range(self.batch_size)])

    def get_total_count(self):
        """
        Get the total count of sequence in the database up to {custom_lenght}
        :return: Total count
        """
        return sum([len(os.listdir(self.videos[i]["input"])) - self.sequence_size for i in range(self.max_video)])

    def get_count_on_video(self, videoid):
        """
        Get the count of sequence on a video
        :param videoid:
        :return: Count
        """
        return len(os.listdir(self.videos[videoid]["input"])) - self.sequence_size

    def get_total_test_count(self):
        """
        Get the total count of sequence on the database after {custom_lenght}
        :return: Count
        """
        return sum([len(os.listdir(self.videos[i]["input"])) for i in range(self.max_video, len(self.videos))])

    def load_imgs(self, imgs):
        """
        Load an image resized to ({size},{size})
        :param imgs: paths to load the images
        :return [N,size,size,channel]
        """
        return np.asarray([cv2.resize(cv2.imread(i), (self.size, self.size)) for i in imgs])


