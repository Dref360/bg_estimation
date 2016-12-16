import csv
from collections import Iterable, OrderedDict

import keras
import numpy as np


def chunks(l, n):
    """Yield successive n-sized chunks from l.
    :param l list to chunk
    :param n size of chunk
    :return Generator from chunk"""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_shape(T):
    """
    Get the shape of a Tensorflow Tensor
    :param T: Tensor
    :return: Shape of the tensor
    """
    return [i.value for i in T.get_shape()]


class CSVLogging():
    """
    Easy to use logger to log list of values
    """
    def __init__(self, fn, head):
        """
        Initialized the logger
        :param fn: filename
        :param head: header ie [head1,...,headn]
        """
        self.fp = open(fn, "w")
        print(",".join(head), file=self.fp)

    def write(self, values):
        """
        Write the list to the file
        :param values: [v1,...,vn]
        :return: None
        """
        print(",".join(values), file=self.fp, flush=True)

    def close(self):
        """
        Close the file
        :return: None
        """
        self.fp.close()


class CSVLogger(keras.callbacks.Callback):
    '''Callback that streams epoch results to a csv file.
    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.
    # Example
        ```python
            csv_logger = CSVLogger('training.log')
            model.fit(X_train, Y_train, callbacks=[csv_logger])
        ```
    Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    '''

    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        super(CSVLogger, self).__init__()

    def on_train_begin(self, logs={}):
        if self.append:
            self.csv_file = open(self.filename, 'a')
        else:
            self.csv_file = open(self.filename, 'w')

    def on_epoch_end(self, epoch, logs={}):
        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(lambda x: str(x), k)))
            else:
                return k

        if not self.writer:
            self.keys = sorted(logs.keys())
            self.writer = csv.DictWriter(self.csv_file, fieldnames=['epoch'] + self.keys)
            self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs={}):
        self.csv_file.close()