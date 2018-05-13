class Affine:

    def __init__(self, weights, bias, optimizer=None):
        self.W = weights
        self.b = bias
        self.op = optimizer

