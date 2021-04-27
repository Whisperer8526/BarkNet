from tensorflow import keras

# After 10 epochs it's overfitting [train_accuracy = 0.97, valid_accuracy = 0.51]

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
