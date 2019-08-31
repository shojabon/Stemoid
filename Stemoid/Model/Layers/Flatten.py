class Flatten:
    #平坦化層
    #Flattening Layer

    def __init__(self):
        self.input_shape = None

    def set_weights(self, weights, bias):
        pass

    def get_size(self):
        return {'type': 'Flatten',
                'size': self.input_shape}
    #出力型取得
    #Get Output Shape
    def get_output_shape(self, input_shape):
        self.input_shape = input_shape
        output_num = 1
        for x in input_shape:
            output_num = output_num * x
        return output_num

    def compile(self):
        pass

    # 　 前方計算処理関数
    #   Forward Propagate Function
    def forward(self, input_data):
        Num, C, With, Hight = input_data.shape
        self.input_shape = input_data.shape
        return input_data.reshape(Num, -1)

    #   逆転伝播計算関数
    #   Back Propagation Calculation Function
    def backward(self, dout):
        return dout.reshape(self.input_shape)