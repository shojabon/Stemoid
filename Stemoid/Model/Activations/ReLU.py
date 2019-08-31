#ReLU活性化関数
#ReLU Activation Function

class ReLU:
    def __init__(self):
        self.mask = None

    # 　 前方計算処理関数
    #   Forward Propagate Function
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    #   逆転伝播計算関数
    #   Back Propagation Calculation Function
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx