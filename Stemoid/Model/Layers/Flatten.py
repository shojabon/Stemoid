class Flatten:

    def __init__(self):
        self.input_shape = None

        self.doutWeights = 0
        self.doutBias = 0

        self.weights = 0
        self.bias = 0

    def set_weights(self, weights, bias):
        pass

    def get_size(self):
        return {'type': 'Flatten',
                'size': self.input_shape}

    def get_output_shape(self, input_shape):
        self.input_shape = input_shape
        output_num = 1
        for x in input_shape:
            output_num = output_num * x
        return output_num

    def compile(self):
        pass

    def forward(self, input_data):
        Num, C, With, Hight = input_data.shape
        self.input_shape = input_data.shape
        return input_data.reshape(Num, -1)

    def backward(self, dout):
        return dout.reshape(self.input_shape)