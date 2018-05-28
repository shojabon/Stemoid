import numpy as np

from Stemoid.Model.Activations.ReLU import ReLU
from Stemoid.Model.Layers.Affine import Affine
from Stemoid.Model.Layers.Convolution import Convolution
from Stemoid.Model.Layers.Input import Input
from Stemoid.Model.ModelBuilder import ModelBuilder
from dataset.mnist import load_mnist
from load_cifar import load_batch

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
data, label = load_batch()
model = ModelBuilder()
model.add(Input((3, 32, 32)))
model.add(Convolution(filter_num=16, filter_shape=(2, 2)))
model.add(Convolution(filter_num=16, filter_shape=(3, 3)))
model.add(Convolution(filter_num=32, filter_shape=(5, 5)))
model.add(Convolution(filter_num=64, filter_shape=(10, 10)))

model.compile()

model.predict(data[0])