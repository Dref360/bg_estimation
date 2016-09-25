import abc

Relu = "relu"


def print_shape(x):
    print(x.get_shape())
    return x


class BaseModel():
    """
    Base model for every model.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, name):
        self.name = name
        self.model = None

    @abc.abstractmethod
    def train_on(self, batch, gt):
        raise NotImplemented

    def get_model(self):
        return self.model
