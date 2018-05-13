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
