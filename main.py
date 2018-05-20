import numpy as np

from Stemoid.Model.Activations.ReLU import ReLU
from Stemoid.Model.Layers.Affine import Affine
from Stemoid.Model.Layers.Input import Input
from Stemoid.Model.ModelBuilder import ModelBuilder
from dataset.mnist import load_mnist
from load_cifar import load_batch

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
model = ModelBuilder()
model.add(Input((784,)))
model.add(Affine(50, activation=ReLU()))
model.add(Affine(10))
model.compile()

data, label = load_batch()
print(data[0])
print(label[0])