import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense
print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

print(x_valid.shape, y_valid.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
import sklearn
from sklearn.preprocessing import StandardScaler
SC = StandardScaler()
X_train =SC.fit_transform(x_train.astype(np.float32).reshape(-1,1)).reshape(-1,28,28,1)
X_test =SC.fit_transform(x_test.astype(np.float32).reshape(-1,1)).reshape(-1,28,28,1)
X_valid =SC.fit_transform(x_valid.astype(np.float32).reshape(-1,1)).reshape(-1,28,28,1)

model = keras.models.Sequential()
model.add(Conv2D(filters=32,kernel_size=3,padding="same",activation="selu",input_shape=(28,28,1)))
model.add(keras.layers.Conv2D(filters=32, kernel_size=3,
                              padding='same',
                              activation='selu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3,
                              padding='same',
                              activation='selu'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3,
                              padding='same',
                              activation='selu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Conv2D(filters=128, kernel_size=3,
                              padding='same',
                              activation='selu'))
model.add(keras.layers.Conv2D(filters=128, kernel_size=3,
                              padding='same',
                              activation='selu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='selu'))
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer = "sgd",
              metrics = ["accuracy"])

model.summary()

logdir = './cnn-selu-callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir,
                                 "fashion_mnist_model.h5")

callbacks = [
             keras.callbacks.EarlyStopping(patience=5,min_delta=1e-3),
             keras.callbacks.ModelCheckpoint(output_model_file,save_best_only=True)]

history = model.fit(X_train,y_train,validation_data=(X_valid,y_valid),epochs=20,callbacks=callbacks)