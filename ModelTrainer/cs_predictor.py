import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.python.framework.dtypes as dtypes

print(f"Numpy {np.__version__}")
print(f"Keras {keras.__version__}")
print(f"Tensorflow {tf.__version__}")

from keras import layers


def train_model():
    # load dataset
    (ds_train, ds_test) = tfds.as_numpy(tfds.load(name="cs_skin_price_data", split=["train", "test"], batch_size=-1))

    # gather training data from dataset
    texture_train, price_train, rarity_train, weapon_train = ds_train["texture"], ds_train["price"], ds_train["rarity"], \
    ds_train["weapon_type"]

    # gather testing data from dataset
    texture_test, price_test, rarity_test, weapon_test = ds_test["texture"], ds_test["price"], ds_test["rarity"], \
    ds_test["weapon_type"]

    # compile training data
    x_train = (texture_train, weapon_train, rarity_train)
    y_train = price_train

    # compile testing data
    x_test = (texture_test, weapon_test, rarity_test)
    y_test = price_test

    # -- create model --

    # set input layers
    texture_input = keras.Input(shape=(512, 512, 4), name="texture")
    weapon_input = keras.Input(batch_shape=(1,), dtype=dtypes.int32, name="weapon_type")
    rarity_input = keras.Input(batch_shape=(1,), dtype=dtypes.int32, name="rarity")

    """
    dense = keras.layers.Dense(64, activation="relu")
    x = dense(texture_input)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    """

    # create mode layers
    # combine alpha channel
    combined_alpha = layers.Dense(units=3, activation="relu")(texture_input)

    # collect features
    conv_layer = layers.ConvLSTM1D(filters=256, kernel_size=(32,))(combined_alpha)

    """
    # create data pool
    pooled_layer = layers.MaxPool1D(pool_size=(4,))(conv_layer)

    # flatten texture
    # flattened = layers.Flatten(pooled_layer)

    # reduce texture to smaller size
    texture_reducer = layers.Dense(units=128, activation="relu")(pooled_layer)

    # reduce input shape of texture
    texture_reducer_2 = layers.Dense(units=1)(texture_reducer)

    # flatten texture
    flattened = layers.Flatten()(texture_reducer_2)

    # reshape data for concat
    weapon_input_reshaped = layers.Reshape((-1,))(weapon_input)
    rarity_input_reshaped = layers.Reshape((-1,))(rarity_input)

    # combine texture with weapon type and rarity
    combined = layers.Concatenate(axis=-1)([weapon_input_reshaped, rarity_input_reshaped, flattened])

    """

    # create output layer
    outputs = layers.Dense(units=1)(conv_layer)

    # compile model
    model = keras.Model(inputs=[texture_input, weapon_input, rarity_input], outputs=outputs, name="cs_predictor_model")
    model.summary()

    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.RMSprop(),
        metrics=["accuracy"]
    )

    # train model
    history = model.fit(x_train, y_train, batch_size=12, epochs=16)

    # test model
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test loss: {test_scores[0]}")
    print(f"Test accuracy: {test_scores[1]}")

    # save model
    model.save("cs_prediction.keras")


def main():
    while True:
        # get user input
        x = input("Please select an option:\n1.) Train\n2.) Predict\n")

        if x == "1":
            train_model()
        elif x == "2":
            pass


if __name__ == "__main__":
    train_model()
