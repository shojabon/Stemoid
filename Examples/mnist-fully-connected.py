#全統合層とドロップアウト層でmnistデータを学習
#Learn mnist Data With Fully-Connected Layers And Dropout Layers

import sys, pathlib
current_dir = pathlib.Path(__file__).resolve().parent
# モジュールのあるパスを追加
sys.path.append( str(current_dir) + '/../' )

from Stemoid.Model.Activations.ReLU import ReLU
from Stemoid.Model.Layers.Affine import Affine
from Stemoid.Model.Layers.Dropout import Dropout
from Stemoid.Model.Layers.Input import Input
from Stemoid.Model.ModelBuilder import ModelBuilder
from Stemoid.Model.Optimizers.Adam import Adam
from Libraries.mnist import load_mnist
import numpy as np

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

learning_rate = 0.001
batch_size = 128
model = ModelBuilder()
model.add(Input(np.array([784])))
model.add(Affine(512, activation=ReLU()))
model.add(Dropout(0.2))
model.add(Affine(512, activation=ReLU()))
model.add(Dropout(0.2))
model.add(Affine(10))
model.compile(optimizer=Adam(lr=learning_rate))

model.execute_learning_session(training_question=x_train,
                               traning_label=t_train,
                               batch_size=128,
                               epoch=2,
                               validation_data=(x_test,t_test))
