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
model.add(Affine(10, activation=ReLU()))
model.compile()

for x in range(200000):
    batch_mask = np.random.choice(100, 100)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    gradient = model.get_gradient(x_batch, t_batch)
    for y in range(len(gradient)):
        model.model[y].influence_weights(gradient[y], learning_rate=0.01)
    if x % 10000 == 0:
        print(model.get_loss(x_test, t_test))
        print(model.get_accuracy(x_test, t_test))
