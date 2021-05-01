from tensorflow import keras

# BarkNet v1.0
# After 10 epochs it's overfitting 
# [train_accuracy = 0.97, valid_accuracy = 0.51]

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[224,224]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(224, kernel_initializer="he_normal", use_bias=False))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Dense(112, kernel_initializer="he_normal", use_bias=False))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Dense(56, kernel_initializer="he_normal", use_bias=False))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Dense(28, kernel_initializer="he_normal", use_bias=False))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Dense(7, kernel_initializer="he_normal", use_bias=False))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Dense(5, activation="softmax"))

# BarkNet v1.2
# slightly different architecture with convolution layers after input. Still overfitting 
# [train_accuracy = 0.97, valid_accuracy = 0.80] 35 epochs
# after initial 20 epochs learning rate has been reduced to 0.0001

from functools import partial

RegularDense = partial(keras.layers.Dense,
                       activation="elu", 
                       kernel_initializer="he_normal",
                       kernel_regularizer=keras.regularizers.l2(0.01))

from tensorflow.keras.optimizers import Nadam

BarkNet1_2.compile(loss="sparse_categorical_crossentropy",
             optimizer=Nadam(learning_rate=0.001),
             metrics=["accuracy"])


BarkNet1_2 = keras.models.Sequential()
BarkNet1_2.add(keras.layers.InputLayer(input_shape=[224,224,3]))
BarkNet1_2.add(keras.layers.BatchNormalization())
BarkNet1_2.add(keras.layers.ZeroPadding2D(padding=3))
BarkNet1_2.add(keras.layers.Conv2D(filters=32, kernel_size=(7,7), data_format='channels_last'))
BarkNet1_2.add(keras.layers.MaxPooling2D())

BarkNet1_2.add(keras.layers.Flatten())
BarkNet1_2.add(keras.layers.BatchNormalization())
BarkNet1_2.add(RegularDense(112))
BarkNet1_2.add(keras.layers.BatchNormalization())
BarkNet1_2.add(RegularDense(56))
BarkNet1_2.add(keras.layers.BatchNormalization())
BarkNet1_2.add(RegularDense(28))
BarkNet1_2.add(keras.layers.BatchNormalization())
BarkNet1_2.add(RegularDense(14))
BarkNet1_2.add(keras.layers.Dropout(0.3))
BarkNet1_2.add(keras.layers.Dense(5, activation="softmax",
                                  kernel_initializer="glorot_uniform"))

# BarkNet v1.3
# This network doesn't show major overfitting symptoms unlike its predecessors
# [train_accuracy = 0.97, valid_accuracy = 0.90] 15 epochs
# Learning rate has been changed every 5 epochs  0.003 -> 0.001 -> 0.0001


from functools import partial

RegularDense = partial(keras.layers.Dense,
                       activation="elu", 
                       kernel_initializer="he_normal",
                       kernel_regularizer=keras.regularizers.l2(0.01))

from tensorflow.keras.optimizers import Nadam

BarkNet1_2.compile(loss="sparse_categorical_crossentropy",
             optimizer=Nadam(learning_rate=0.0001),
             metrics=["accuracy"])

BarkNet1_2 = keras.models.Sequential()
BarkNet1_2.add(keras.layers.InputLayer(input_shape=[224,224,3]))
BarkNet1_2.add(keras.layers.BatchNormalization())
BarkNet1_2.add(keras.layers.ZeroPadding2D(padding=3))
BarkNet1_2.add(keras.layers.Conv2D(filters=32, kernel_size=(7,7), data_format='channels_last'))
BarkNet1_2.add(keras.layers.MaxPooling2D())
BarkNet1_2.add(keras.layers.Conv2D(filters=32, kernel_size=(5,5), data_format='channels_last', padding="same"))
BarkNet1_2.add(keras.layers.MaxPooling2D())
BarkNet1_2.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), data_format='channels_last', padding="same"))
BarkNet1_2.add(keras.layers.MaxPooling2D())

BarkNet1_2.add(keras.layers.Flatten())
BarkNet1_2.add(keras.layers.BatchNormalization())
BarkNet1_2.add(RegularDense(28))
BarkNet1_2.add(keras.layers.BatchNormalization())
BarkNet1_2.add(RegularDense(28))
BarkNet1_2.add(keras.layers.BatchNormalization())
BarkNet1_2.add(RegularDense(14))
BarkNet1_2.add(keras.layers.BatchNormalization())
BarkNet1_2.add(RegularDense(7))
BarkNet1_2.add(keras.layers.Dropout(0.3))
BarkNet1_2.add(keras.layers.Dense(5, activation="softmax",
                                  kernel_initializer="glorot_uniform"))
