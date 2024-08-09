import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf


# define the sorting key for importing pictures
def extract_integer(filename):
    return int(filename.split('.')[0])


datagap = np.array([pd.read_csv(os.path.join("data/gap",elem)).to_numpy() for elem in sorted(os.listdir('data/gap'),key=extract_integer)])
size, dimension = datagap.shape[0] , datagap.shape[1]
datagap = datagap.reshape((size,dimension))
gap_data = np.array([[0,elem] for elem in datagap])

datagapless = np.array([pd.read_csv(os.path.join("data/gapless",elem)).to_numpy() for elem in sorted(os.listdir('data/gapless'), key = extract_integer)])
size, dimension = datagapless.shape[0] , datagapless.shape[1]
datagapless = datagapless.reshape((size,dimension))
gapless_data = np.array([[1,elem] for elem in datagapless])

X = np.vstack(np.hstack((gap_data[:,1],gapless_data[:,1])))
Y = np.hstack([gap_data[:,0],gapless_data[:,0]])

x_train ,x_test ,y_train ,y_test = train_test_split(X.reshape((len(X),241,241)), Y ,test_size=0.80)
x_train , y_train = tf.convert_to_tensor(x_train ,dtype = tf.float32), tf.convert_to_tensor(y_train ,dtype = tf.float32)
x_test , y_test = tf.convert_to_tensor(x_test ,dtype = tf.float32), tf.convert_to_tensor(y_test ,dtype = tf.float32)
x_train.shape


def vgg16_like_model(input_shape=(241,241,1), num_classes=2):
    model = tf.keras.models.Sequential()
    # Block 1
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))
    # Block 2
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))

    # Block 3
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))
    # Block 4
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))
    # Block 5
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))
    # Fully connected layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1000, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    return model



model = vgg16_like_model()
rlronp = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.0001*np.exp(-epoch/60))
estop=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=32 , callbacks=[rlronp, estop],shuffle = 'True')

