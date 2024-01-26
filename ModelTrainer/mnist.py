import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
import numpy as np

print(f"Numpy {np.__version__}")
print(f"Keras {keras.__version__}")
print(f"Tensorflow {tf.__version__}")

from keras import layers

inputs = keras.Input(shape=(784,))
dense = keras.layers.Dense(64, activation="relu")
x = dense(inputs)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
model.summary()

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"]
)

history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print(f"Test loss: {test_scores[0]}")
print(f"Test accuracy: {test_scores[1]}")

model.save("mnist.keras")

img = keras.utils.img_to_array(keras.utils.load_img("./MNIST/mnist-1.png", color_mode="grayscale"))

prediction = model(img.reshape((1, 784)))

print(np.argmax(prediction))