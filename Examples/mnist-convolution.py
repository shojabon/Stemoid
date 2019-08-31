import sys, pathlib
current_dir = pathlib.Path(__file__).resolve().parent
# モジュールのあるパスを追加
sys.path.append( str(current_dir) + '/../' )

import numpy as np

#畳み込みレイヤーを使ったモデルでmnistデータを学習
#Learn mnist Data With Convolution Layers
from Stemoid.Model.Activations.ReLU import ReLU
from Stemoid.Model.Layers.Affine import Affine
from Stemoid.Model.Layers.Convolution import Convolution
from Stemoid.Model.Layers.Flatten import Flatten
from Stemoid.Model.Layers.Input import Input
from Stemoid.Model.ModelBuilder import ModelBuilder
from Stemoid.Model.Optimizers.Adam import Adam
from Libraries.mnist import load_mnist
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

learning_rate = 0.001
batch_size = 128

#答えを784 から 1x28x28の2+1次元(2次元の画像情報と1次元の色情報)に変換
x_train = x_train.reshape(60000, 1, 28, 28)
x_test = x_test.reshape(10000, 1, 28, 28)

model = ModelBuilder()
model.add(Input(np.array([1,28,28])))
model.add(Convolution(filter_num=10, filter_shape=np.array([20,20]), activation=ReLU()))
model.add(Flatten())
model.add(Affine(512))
model.add(Affine(10))
model.compile(optimizer=Adam(lr=learning_rate))

model.execute_learning_session(training_question=x_train,
                               traning_label=t_train,
                               batch_size=128,
                               epoch=2,
                               validation_data=(x_test,t_test))
