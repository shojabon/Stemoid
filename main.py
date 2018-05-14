import numpy as np

from Stemoid import *
from Stemoid.Model.Activations.ReLU import ReLU
from Stemoid.Model.Layers.Affine import Affine
from Stemoid.Model.Layers.Input import Input
from Stemoid.Model.ModelBuilder import ModelBuilder


model = ModelBuilder()
model.add(Input((1,)))
model.add(Affine(50, activation=ReLU()))
model.add(Affine(10, activation=ReLU()))

model.compile()
a = np.arange(-10, 10).reshape(1, 20)
print(a.argmax(axis=1))