class Input:

    def __init__(self, size_shape, activation=None):
        self.size = size_shape
        self.activation = activation
        pass

    def forward(self, x):
        if x.shape != self.size:
            print('Input shape not the same')
            return
        return x

    def get_size(self):
        s = list(self.size)
        if len(s) == 1:
            return s[0]
        return s

    def backward(self, dout):
        return dout

    def compile(self):
        pass