#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv('../data/google-5yr.csv')[['Date', 'Close']]

class StockPrediction:
    def __init__(self, data):
        self.data = data
        self.X, self.y = [],[]
        self.back_days = 30 # 30
        self.epochs = 300 # 300
        self.total_set_size = len(self.data)
        self.train_set_size = 800
        self.test_test_size = self.total_set_size - self.train_set_size - self.back_days - 1
    
    def preprocess(self):
        self.scaler = MinMaxScaler()
        cl = self.data.reshape(self.total_set_size, 1)
        self.data = self.scaler.fit_transform(cl)
    
    def processData(self):
        for i in range(self.total_set_size - self.back_days - 1):
            self.X.append(self.data[i:(i + self.back_days), 0])
            self.y.append(self.data[(i + self.back_days), 0])
        self.X, self.y = np.array(self.X), np.array(self.y)
    
    def split(self):
        self.X_train, self.X_test = self.X[:self.train_set_size], self.X[self.train_set_size:]
        self.y_train, self.y_test = self.y[:self.train_set_size], self.y[self.train_set_size:]
        #return self.X_train, self.X_test, self.y_train, self.y_test
    
    def createModel(self):
        model = Sequential()
        model.add(LSTM(256, input_shape=(self.back_days, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self):
        self.preprocess()
        self.processData()
        self.split()
        self.model = self.createModel()
        
        self.X_train = self.X_train.reshape((self.train_set_size, self.back_days, 1))
        self.X_test = self.X_test.reshape((self.test_test_size, self.back_days, 1))
        #, validation_data=(self.X_test, self.y_test)
        self.history = self.model.fit(self.X_train, self.y_train, epochs=self.epochs, shuffle=False)
    def draw(self):
        #plt.plot(self.history.history['loss'])
        #plt.plot(self.history.history['val_loss'])
        
        Xt = self.model.predict(self.X_test)
        plt.plot(self.scaler.inverse_transform(self.y_test.reshape(-1,1)), label='Real stock price')
        plt.plot(self.scaler.inverse_transform(Xt), label='Predicted stock price')
        

if __name__ == '__main__':
    data = df.values[:, 1]
    #X_train, X_test, y_train, y_test = stock.split()
    stock = StockPrediction(data)
    stock.train()
    stock.draw()    
