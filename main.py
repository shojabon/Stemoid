import numpy as np

from Stemoid import *
from Stemoid.Model.Activations.ReLU import ReLU
from Stemoid.Model.Layers.Affine import Affine
from Stemoid.Model.Layers.Input import Input
from Stemoid.Model.ModelBuilder import ModelBuilder
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
model = ModelBuilder()
model.add(Input((784,)))
model.add(Affine(50, activation=ReLU()))
model.add(Affine(10))
model.compile()
print(model.get_loss(x_train[0], t_train[0]))