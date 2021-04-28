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

# BarkNet v2.0
# slightly different architecture with convolution layers after input. Still overfitting 
# [train_accuracy = 0.93, valid_accuracy = 0.64]


BarkNet2 = keras.models.Sequential()
BarkNet2.add(keras.layers.InputLayer(input_shape=[224,224,3]))
BarkNet2.add(keras.layers.ZeroPadding2D(padding=3))
BarkNet2.add(keras.layers.Conv2D(filters=32, kernel_size=(7,7), data_format='channels_last'))
BarkNet2.add(keras.layers.MaxPooling2D())
BarkNet2.add(keras.layers.ZeroPadding2D())
BarkNet2.add(keras.layers.Conv2D(filters=32, kernel_size=(5,5), data_format='channels_last', padding='same'))
BarkNet2.add(keras.layers.MaxPooling2D())

BarkNet2.add(keras.layers.Flatten())
BarkNet2.add(keras.layers.Dense(56, kernel_initializer="he_normal", use_bias=False))
BarkNet2.add(keras.layers.BatchNormalization())
BarkNet2.add(keras.layers.Activation("relu"))
BarkNet2.add(keras.layers.Dropout(0.1))

BarkNet2.add(keras.layers.Dense(28, kernel_initializer="he_normal", use_bias=False))
BarkNet2.add(keras.layers.BatchNormalization())
BarkNet2.add(keras.layers.Activation("relu"))
BarkNet2.add(keras.layers.Dropout(0.1))

BarkNet2.add(keras.layers.Dense(14, kernel_initializer="he_normal", use_bias=False))
BarkNet2.add(keras.layers.BatchNormalization())
BarkNet2.add(keras.layers.Activation("relu"))
BarkNet2.add(keras.layers.Dropout(0.1))

BarkNet2.add(keras.layers.Dense(14, kernel_initializer="he_normal", use_bias=False))
BarkNet2.add(keras.layers.BatchNormalization())
BarkNet2.add(keras.layers.Activation("relu"))
BarkNet2.add(keras.layers.Dropout(0.1))

BarkNet2.add(keras.layers.Dense(7, kernel_initializer="he_normal", use_bias=False))
BarkNet2.add(keras.layers.BatchNormalization())
BarkNet2.add(keras.layers.Activation("relu"))

BarkNet2.add(keras.layers.Dense(7, kernel_initializer="he_normal", use_bias=False))
BarkNet2.add(keras.layers.BatchNormalization())
BarkNet2.add(keras.layers.Activation("relu"))

BarkNet2.add(keras.layers.Dropout(0.2))
BarkNet2.add(keras.layers.Dense(5, activation="softmax"))
