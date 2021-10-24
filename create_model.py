import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from utils import load_data

#Options: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgusted, Surprised
emotions_to_observe = ['Neutral', 'Happy','Angry']

#Load the data and extract features for each sound file from the RAVDESS dataset
x,y=load_data("/home/moe/Desktop/projects/sound_freq/speech_emotion_app/speech_data/Actor_*/*.wav",
emotions_to_observe)

#Initialize multilayer perceptron model (alpha: L2 regularization parameter,
# bath_size: Size of minibatches for stochastic optimizers deafault --> 200)
# max iter: Maximum number of iterations. The solver iterates until convergence or this number of iterations.
model=MLPClassifier(batch_size=256, max_iter=500)

#split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=24)

#fit the model
model.fit(x_train,y_train)

#predict the emotion using testing features
y_pred=model.predict(x_test)

#calculate accuracy of predictions
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print(f"Accuracy of Model {accuracy*100}")
print(f"Accuracy of Random Guessing {1/len(emotions_to_observe)*100}")

# Save model into .model binary to use in application
pickle.dump(model, open("model.model", "wb"))

from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt
# load dataset
# series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# # split dataset
# X = series.values
# train, test = X[1:len(X)-7], X[len(X)-7:]
# train autoregression
num_samples = 9
model = AutoReg(X, lags=29)
model_fit = model.fit()
coef = model_fit.params
# walk forward over time steps in test
history = train[len(train)-num_samples:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-num_samples,length)]
    yhat = coef[0]
for d in range(num_samples):
    yhat += coef[d+1] * lag[num_samples-d-1]
    obs = test[t]
predictions.append(yhat)
# history.append(obs)
print('predicted=%f, expected=%f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot
# pyplot.plot(test)
# pyplot.plot(predictions, color='red')
# pyplot.show()


window = 9
model = AutoReg(train, lags=9)
model_fit = model.fit()
coef = model_fit.params
# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)


model = AutoReg(train, lags=9)
model_fit = model.fit()
# print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

model = AutoReg(train, lags=9)
model_fit = model.fit()

predictions = model_fit.predict(start=len(train), end=len(train + len(test)-1, dynamic =False)
rmse = sqrt(mean_squared_error(test, predictions))

print('Test RMSE: %.3f' %rmse)
