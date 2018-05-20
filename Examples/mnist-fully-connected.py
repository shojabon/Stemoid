import numpy as np

from Stemoid.Model.Activations.ReLU import ReLU
from Stemoid.Model.Layers.Affine import Affine
from Stemoid.Model.Layers.Input import Input
from Stemoid.Model.ModelBuilder import ModelBuilder
from Stemoid.Model.Optimizers.SGD import SGD
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

learning_rate = 0.1
batch_size = 100

model = ModelBuilder()
model.add(Input((784,)))
model.add(Affine(50, activation=ReLU()))
model.add(Affine(10))
model.compile(optimizer=SGD(lr=learning_rate))

for x in range(200000):
    batch_mask = np.random.choice(x_train.shape[0], batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    model.learn(x_batch, t_batch)
    if x % 1000 == 0:
        print(model.get_accuracy(x_test, t_test))