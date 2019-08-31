import random

import numpy as np

from Libraries.SBar import SBar
from Stemoid.Model.LossFunctions.Softloss import SoftLoss
from Stemoid.Model.Optimizers.SGD import SGD

#モデル作成
#Model Builder

class ModelBuilder:

    def __init__(self):
        #モデル型記録
        #Model Shape
        self.model = []
        self.model_shape = []
        self.model_output_shape = []
        self.gradient_model = []
        self.input = None
        self.first_done = False
        self.loss = None
        self.optimizer = None

        #学習データ
        #Traning Data
        self.train_data = None
        self.train_label = None
        self.train_label_label = None

        #学習記録
        #Traning Record
        self.epoch = 0
        self.iteration = 0
        self.hidden_iteration = 0

    #レイヤー追加
    #Add Layer
    def add(self, layer):
        if self.first_done:
            self.model.append(layer)
        if not self.first_done:
            self.input = layer
            self.first_done = True
        return self

    #モデルをビルド
    #Builder Model
    def compile(self, optimizer=SGD(), loss=SoftLoss()):
        input_shape = self.input.get_output_shape()
        self.model_output_shape.append(input_shape)
        for x in range(len(self.model)):
            input_shape = self.model[x].get_output_shape(input_shape)
            self.model_output_shape.append(input_shape)


        self.model_shape.append(self.input.get_size())
        for x in self.model:
            self.model_shape.append(x.get_size())

        for x in range(len(self.model)):
            if hasattr(self.model[x], 'weights'):
                self.gradient_model.append(x)

        for x in range(len(self.model_shape) - 1):
            self.model[x].compile()
        self.loss = loss
        self.optimizer = optimizer

    #モデルを使い結果を予測
    #Predict Answer With The Model
    def predict(self, input_data):
        input_data = self.input.forward(input_data)
        for x in range(len(self.model_shape) - 1):
            input_data = self.model[x].forward(input_data)
        return input_data

    #訓練データーから損失を計算
    #Calculate Loss From Traning Data
    def get_loss(self, input_data, label):
        out = self.predict(input_data)
        return self.loss.forward(out, label)

    #入力データから勾配を計算
    #Calculate Gradiant From Input Data
    def get_gradient(self, input_data, label):
        self.get_loss(input_data, label)
        dout = self.loss.backward()
        lista = list(self.model)
        lista.reverse()
        out = {}
        for x in lista:
            dout = x.backward(dout)
        lista.reverse()
        for x in self.gradient_model:
            out[x] = {'weights': lista[x].doutWeights,
                        'bias': lista[x].doutBias}
        return out

    #入力データと答えから正解率を取得
    #Get Accuracy From Input Data And Answer
    def get_accuracy(self, input_data, answer):
        y = self.predict(input_data)
        y = np.argmax(y, axis=1)
        if answer.ndim != 1:
            t = np.argmax(answer, axis=1)
        accuracy = np.sum(y == t) / float(input_data.shape[0])
        return accuracy

    #入力データと答えから1ステップ学習する
    #Learn One Step From Input Data And Output Data
    def learn(self, input_data, label):
        gradient = self.get_gradient(input_data, label)
        self.optimizer.update(self.model, gradient, self.gradient_model)

    #データを小分けして入力数分のデータで１回学習する
    #Split Training Data Into Small Sets And Learn The Data With An Amount From It Once
    def learn_epoch(self, batch_size=100):
        if self.train_label_label.size < batch_size:
            self.epoch += 1
            self.train_label_label = np.arange(0, self.train_label.shape[0])
        mask = np.random.choice(self.train_label_label.size, batch_size)
        self.iteration += batch_size
        self.learn(self.train_data[mask], self.train_label[mask])
        self.train_label_label = np.delete(self.train_label_label, mask)

    #データをランダムに並べ小分けにして入力数分１回学習する
    #Arrange Traning Data Randomly And Learn The Data With An Amount From It Once
    def learn_random(self, batch_size=100):
        if self.hidden_iteration >= self.train_label.size:
            self.epoch += 1
            self.hidden_iteration = 0
        mask = np.random.choice(self.train_label_label.size, batch_size)
        self.iteration += batch_size
        self.hidden_iteration += batch_size
        self.learn(self.train_data[mask], self.train_label[mask])

    #学習データをメモリにセットする
    #Load The Traning Data To Memory
    def set_training_data(self, data, label):
        self.train_data = data
        self.train_label = label
        self.train_label_label = np.arange(0, label.shape[0])

    #セットされた学習データを一周学習する
    #Learn Whole Training Data Once
    def learn_whole_epoch(self, batch_size):
        index = [x for x in range(len(self.train_data))]
        random.shuffle(index)
        iterations = len(index)//batch_size
        remaining_iterations = (len(index)/batch_size-iterations)*batch_size
        for x in range(iterations):
            mask = index[x*128:(x+1)*128-1]
            xmask = self.train_data[mask]
            tmask = self.train_label[mask]
            self.learn(xmask, tmask)
            iteration_count = str((x+1)*128) + '/' + str(len(self.train_label))
            acc = round(self.get_accuracy(xmask, tmask).item()*100, 2)
            print('Iteration ' + iteration_count + '  ' + SBar().render_bar(x/iterations, 5) + ' Training Data Accuracy ' + str(acc) + '%')
        if remaining_iterations != 0:
            mask = index[iterations * 128: len(self.train_label)]
            xmask = self.train_data[mask]
            tmask = self.train_label[mask]
            self.learn(xmask, tmask)
            acc = round(self.get_accuracy(xmask, tmask).item()*100, 2)
            print('Iteration ' + str(str(len(self.train_label)) + '/' + str(len(self.train_label))) + '  ' + SBar().render_bar(1, 5) + ' Training Data Accuracy ' + str(acc) + '%')

    #学習セッションを開始する
    #Start Learning Session
    def execute_learning_session(self, training_question, traning_label, batch_size, epoch, validation_data):
        self.set_training_data(training_question, traning_label)
        for x in range(1, epoch+1):
            print('Starting Epoch ' + str(x) + '/' + str(epoch))
            self.learn_whole_epoch(batch_size)
            trainAccuracy = round(self.get_accuracy(self.train_data, self.train_label).item()*100, 2)
            testAccuracy = round(self.get_accuracy(validation_data[0], validation_data[1]).item()*100, 2)
            print('Epoch ' + str(x) + '/' + str(epoch) + '  Training Accuracy:' + str(trainAccuracy) + '%' + ' Test Accuracy:' + str(testAccuracy) + '%')