class GeneratorLoop(object):
    def __init__(self, f):
        self.fun = f

    def __call__(self, *args):
        while True:
            yield from self.fun(*args)

    def __iter__(self):
        while True:
            yield from self.fun()