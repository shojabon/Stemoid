import numpy as np

#ドロップアウト層
#Droupout Layer

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        #       レイヤーパラメーター
        #       Variables
        self.dropout_ratio = dropout_ratio
        #       入力データ記録
        #       Records Of Input Data
        self.mask = None
        self.input_shape = None

    # 　 前方計算処理関数
    #   Forward Propagate Function
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    #   逆転伝播計算関数
    #   Back Propagation Calculation Function
    def backward(self, dout):
        return dout * self.mask
    #出力型取得
    #Get Output Shape
    def get_output_shape(self, input_shape):
        self.input_shape = input_shape
        return input_shape


    def get_size(self):
        return {'type':'Dropout',
                'size':(self.dropout_ratio, self.input_shape),}
    #レイヤーコンパイル
    #Compile The Layer With The Variable
    def compile(self):
        pass