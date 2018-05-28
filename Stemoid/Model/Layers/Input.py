class Input:

    def __init__(self, size_shape):
        self.size = size_shape
        pass

    def forward(self, x):
        if x.ndim == 2:
            if x.shape[1] != self.size[0]:
                print('Input shape not the same')
                return
            return x
        if x.shape != self.size:
            print('Input shape not the same')
            return
        return x

    def get_size(self):
        return {'type': 'Input',
                'size': self.size}

    def get_output_shape(self):
        if len(self.size) is 1:
            return self.size[0]
        return self.size

    def compile(self):
        pass