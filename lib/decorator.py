class GeneratorLoop(object):
    """
    Decorator to create an infinite generator from a finite one
    """
    def __init__(self, f):
        """
        Initialize
        :param f: function that create the generator
        """
        self.fun = f

    def __call__(self, *args):
        while True:
            yield from self.fun(*args)

    def __iter__(self):
        while True:
            yield from self.fun()