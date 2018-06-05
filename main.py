import numpy as np

from Stemoid.Model.Layers.MaxPooling import MaxPooling
from Stemoid.Model.Layers.Dropout import Dropout
from Stemoid.Model.Layers.Affine import Affine
from Stemoid.Model.Layers.BatchNormalization import BatchNormalization
from Stemoid.Model.Layers.Convolution import Convolution
from Stemoid.Model.Layers.Flatten import Flatten
from Stemoid.Model.Layers.Input import Input
from Stemoid.Model.ModelBuilder import ModelBuilder
from Stemoid.Model.Optimizers.Adam import Adam
from dataset.mnist import load_mnist
from load_cifar import load_batch


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
data, label = load_batch()
model = ModelBuilder()
model.add(Input((3, 32, 32)))
model.add(Convolution(filter_num=30, filter_shape = (5, 5), padding=0, activation=BatchNormalization(1, 0)))
model.add(MaxPooling(5, 5))
model.add(Flatten())
model.add(Affine(100))
model.add(Affine(10))
model.compile(optimizer=Adam(lr=0.001))

model.set_training_data(x_train, t_train)
batch_size = 64
for x in range(200000):
    batch_mask = np.random.choice(data.shape[0], batch_size)
    x_batch = data[batch_mask]
    t_batch = label[batch_mask]
    model.learn(x_batch.reshape(batch_size, 3, 32, 32), t_batch)
    if x % 10 == 0:
        print(model.get_loss(x_batch.reshape(batch_size, 3, 32, 32), t_batch), model.get_accuracy(x_batch.reshape(batch_size, 3, 32, 32), t_batch))