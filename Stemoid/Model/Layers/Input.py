import collections

#入力層
#Input Layer

class Error(Exception):
    pass
class Input:

    def __init__(self, size_shape):
        self.size = size_shape

    # 　 前方計算処理関数
    #   Forward Propagate Function
    def forward(self, x):
        if x.ndim is len(self.size) + 1:
            xdim = list(x.shape)
            selfdim=list(self.size)
            xdim.pop(0)
            compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
            if compare(xdim, selfdim) is False:
                raise Error('Input shape doesn\'t match')
            return x
        if x.ndim is len(self.size):
            if x.shape is self.size:
                return x
            raise Error('Input shape doesn\'t match')
        raise Error('Input shape doesn\'t match')


    def get_size(self):
        return {'type': 'Input',
                'size': self.size}
    #出力型取得
    #Get Output Shape
    def get_output_shape(self):
        if len(self.size) is 1:
            return self.size[0]
        return self.size

    def compile(self):
        pass