# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 19:45:03 2020

@author: nesto
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from joblib import dump


dfa = pd.read_excel('Datos.xlsx')
dfa2 = dfa.values


training_set = dfa2[:, 1:2]

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []

plt.plot(training_set_scaled)
plt.show()




nHoras = 40

for i in range(nHoras, len(training_set_scaled)-nHoras+1):
    X_train.append(training_set_scaled[i-nHoras:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


Red = Sequential()

Red.add(LSTM(units = 150, return_sequences = True, input_shape = (X_train.shape[1], 1)))
Red.add(Dropout(0.2)) 

Red.add(LSTM(units = 100, return_sequences = True))
Red.add(Dropout(0.2))

Red.add(LSTM(units = 100, return_sequences = True))
Red.add(Dropout(0.2))

Red.add(LSTM(units = 50))
Red.add(Dropout(0.2))


Red.add(Dense(units = 1))


Red.compile(optimizer = 'adam', loss = 'mean_squared_error')


TH=Red.fit(X_train, y_train, epochs = 1, batch_size = 32)

print("trainning loss: ", TH.history['loss'])
plt.plot(TH.history['loss'])
plt.show()




predicted_price = Red.predict(X_train)

plt.plot(y_train)
plt.plot(predicted_price)
plt.legend()
plt.show()

Red.save('redP2')
dump(sc,'std_scaler.bin', compress = True)























































