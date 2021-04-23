from tensorflow import keras

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[224,224]))
model.add(keras.layers.Dense(224, activation="relu")) #consider changing to SeLU
model.add(keras.layers.Dense(224, activation="relu")) 
model.add(keras.layers.Dense(112, activation="relu"))
model.add(keras.layers.Dense(112, activation="relu"))
model.add(keras.layers.Dense(112, activation="relu"))
model.add(keras.layers.Dense(5, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy",
             optimizer="adam",
             metrics=["accuracy"])

import numpy as np
import os
X = np.load(os.path.join(directory, "image_arrays_bw.npy"))
y = np.load(os.path.join(directory, "image_labels.npy"))

y= y-1                    #adjusting labels

X_train = X[:2100, :]     #train-test split
X_test = X[2100:, :]
y_train = y[:2100]
y_test = y[2100:]

history = model.fit(X_train, y_train, epochs=100,
                   batch_size=32, validation_split=0.1)
