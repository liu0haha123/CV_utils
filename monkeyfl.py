import numpy
import tensorflow as tf
import tensorboard
from tensorflow import keras
import PIL
import pandas  as pd
import matplotlib.pyplot as plt
import sklearn



train_dir = "training/training"
valid_dir = "validation/validation"
label_dir = "monkey_labels.txt"

labels = pd.read_csv(label_dir,header=0)
WIDTH = 128
HEIGHT = 128
CHANNEL = 3
batch_size=32
num_classes = 10

train_data_generator = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)
train_generator = train_data_generator.flow_from_directory(directory=train_dir,
                                                   target_size=(HEIGHT,WIDTH),batch_size=batch_size,                                                   seed = 7,
                                                   shuffle = True,
                                                   class_mode = "categorical")
valid_data_generator = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)
valid_generator = valid_data_generator.flow_from_directory(directory=valid_dir,
                                                   target_size=(HEIGHT,WIDTH),batch_size=batch_size,                                                   seed = 7,
                                                   shuffle = True,
                                                   class_mode = "categorical"
)
train_num = train_generator.samples
valid_num = valid_generator.samples
print(train_num, valid_num)

model = keras.models.Sequential()
from tensorflow.keras.layers import Conv2D,Dense,Dropout,MaxPool2D

model.add(Conv2D(filters=32,kernel_size=3,padding="same",activation="relu",input_shape=[WIDTH,HEIGHT,CHANNEL]))
model.add(Conv2D(filters=32,kernel_size=3,padding="same",activation="relu"))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=3, padding='same',
                        activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, padding='same',
                        activation='relu'))
model.add(MaxPool2D(pool_size=2))

model.add(Conv2D(filters=128, kernel_size=3, padding='same',
                        activation='relu'))
model.add(Conv2D(filters=128, kernel_size=3, padding='same',
                        activation='relu'))
model.add(MaxPool2D(pool_size=2))
model.add(keras.layers.Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(num_classes,activation="softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=['accuracy'])
model.summary()


epochs = 10

history = model.fit_generator(train_generator,steps_per_epoch=train_num//batch_size,epochs=epochs,
                              validation_data=valid_generator,validation_steps=valid_num//batch_size)


def plot_learning_curves(history, label, epcohs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_' + label] = history.history['val_' + label]
    pd.DataFrame(data).plot(figsize=(8, 5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.show()


plot_learning_curves(history, 'acc', epochs, 0, 1)
plot_learning_curves(history, 'loss', epochs, 1.5, 2.5)