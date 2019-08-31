#全統合層とドロップアウト層でmnistデータを学習
#Learn mnist Data With Fully-Connected Layers And Dropout Layers

#ランダムなテストデータから学習したモデルを使って推測
#Use Random Test Data And Predict The Answer Using The Model That Was Just Trainned
import random

from Stemoid.Model.Activations.ReLU import ReLU
from Stemoid.Model.Layers.Affine import Affine
from Stemoid.Model.Layers.Dropout import Dropout
from Stemoid.Model.Layers.Input import Input
from Stemoid.Model.ModelBuilder import ModelBuilder
from Stemoid.Model.Optimizers.Adam import Adam
from Libraries.mnist import load_mnist
import numpy as np
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

def plot_data(ind):
    print('++++++++++++++++++++++++++')
    ar = []
    for x in range(28):
        a = ''
        for y in range(28):
            i = x_test[ind].reshape(28, 28)[x][y]
            if i <= 0.7:
                a += ' '
            else:
                a += '*'
        ar.append(a)
    for x in ar:
        print(x)
    print('++++++++++++++++++++++++++')



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

model.execute_learning_session(training_question=x_train,
                               traning_label=t_train,
                               batch_size=128,
                               epoch=1,
                               validation_data=(x_test,t_test))

print('Predicting Image')
i = random.randint(0, 10000)
plot_data(i)
predict = model.predict(np.array([x_test[i]]))[0]
print('Predicted Number ' + str(np.where(predict == predict.max())[0][0]))