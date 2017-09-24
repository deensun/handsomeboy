# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import quandl as qd
auth_tok = "e78FGuFQqh1yxDcZxigu"
start_date = "2015-2-1"
end_date = "2017-8-31"
authtoken=auth_tok
data1 = qd.get("WIKI/AMZN", trim_start = start_date, trim_end = end_date, authtoken=auth_tok)
allN=len(data1)
volumeP = data1['Volume']
closeP = data1['Close']
highP = data1['High']
lowP = data1['Low']
openP = data1['Open']


#import the training set
start =time.clock()
training_set = np.column_stack([closeP,highP,lowP,volumeP])
#Feature scaling
# 1. Normalization 2.Standarlisation(use this)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

# Getting the inputs and the outputs 
x_train =  training_set[0:allN-1]
y_train = training_set[1:allN,0:1]

#reshaping create time step
# observations，timestep，feature
#allN-1 the observations, 1 is timestep,4 is features of data
X_train = np.reshape(x_train,(allN-1,1,4))
#X_train = training_set[0:allN-1]
# Part 2 - Building the RNN
#importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

#Adding the input layer and LSTM layer
# 4 is how many memory units, second 4 is number of features of input data
regressor.add(LSTM(units = 4,activation = 'sigmoid',input_shape =(None,4)))

#Adding the output layer
regressor.add(Dense(units = 1))

#Compile the RNN
regressor.compile(optimizer ='adam' ,loss = 'mean_squared_error')

#Fitting the RNN to the training data
regressor.fit(X_train,y_train,batch_size =32, nb_epoch = 200)

# Make the prediction and visualization
inputs = X_train
predicted_stock_price = regressor.predict(inputs)

real_stock_price = np.reshape(np.array(closeP[1:allN]),(len(np.array(closeP[1:allN])),1))
real_stock_price = sc.fit_transform(real_stock_price)
real_stock_price = sc.inverse_transform(real_stock_price)
predicted_stock_price= sc.inverse_transform(predicted_stock_price)
predicted_stock = predicted_stock_price[0:allN-1]

plt.plot(real_stock_price,color = 'red',label = 'Real_Amazon_Stock_Price')
plt.plot(predicted_stock,color =  'blue',label = 'Predicted_Amazon_Stock_Price')
plt.title('Amazon_Stock_Price Prediction')
plt.xlabel('Time')
plt.ylabel('Amazon_Price')
plt.legend()
plt.show()
time.clock()-start

































