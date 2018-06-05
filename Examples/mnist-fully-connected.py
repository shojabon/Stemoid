import numpy as np

from Stemoid.Model.Activations.ReLU import ReLU
from Stemoid.Model.Layers.Affine import Affine
from Stemoid.Model.Layers.Dropout import Dropout
from Stemoid.Model.Layers.Input import Input
from Stemoid.Model.ModelBuilder import ModelBuilder
from Stemoid.Model.Optimizers.Adam import Adam
from Stemoid.Model.Optimizers.SGD import SGD
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

learning_rate = 0.001
batch_size = 128

model = ModelBuilder()
model.add(Input((784,)))
model.add(Affine(512, activation=ReLU()))
model.add(Dropout(0.2))
model.add(Affine(512, activation=ReLU()))
model.add(Dropout(0.2))
model.add(Affine(10))
model.compile(optimizer=Adam(lr=learning_rate))

acc = []
loss = []
itera = []

model.set_training_data(x_train, t_train)
for x in range(100):
    model.learn_random(batch_size)
    if x % 1 == 0:
        ac = model.get_accuracy(x_test, t_test)
        acc.append(ac)
        loss.append(model.get_loss(x_test, t_test))
        itera.append(model.iteration)
        print(ac, model.epoch, model.iteration)
plt.plot(itera, acc)
plt.plot(itera, loss)
plt.title('Random Epoch Runner')
plt.show()