import numpy as np

from Stemoid.Model.Activations.ReLU import ReLU
from Stemoid.Model.Layers.Affine import Affine
from Stemoid.Model.Layers.Convolution import Convolution
from Stemoid.Model.Layers.Flatten import Flatten
from Stemoid.Model.Layers.Input import Input
from Stemoid.Model.ModelBuilder import ModelBuilder
from Stemoid.Model.Optimizers.AdaGrad import AdaGrad
from Stemoid.Model.Optimizers.Adam import Adam
from dataset.mnist import load_mnist
from load_cifar import load_batch


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
data, label = load_batch()
model = ModelBuilder()
model.add(Input((3, 32, 32)))
model.add(Convolution(filter_num=30, filter_shape=(5, 5)))
model.add(Flatten())
model.add(Affine(50))
model.add(Affine(10))
model.compile(optimizer=AdaGrad(lr=0.01))

batch_size = 100
for x in range(200000):
    batch_mask = np.random.choice(data.shape[0], batch_size)
    x_batch = data[batch_mask]
    t_batch = label[batch_mask]
    model.learn(x_batch.reshape(batch_size, 3, 32, 32), t_batch)
    if x % 10 == 0:
        print(model.get_loss(x_batch.reshape(batch_size, 3, 32, 32), t_batch), model.get_accuracy(x_batch.reshape(batch_size, 3, 32, 32), t_batch))