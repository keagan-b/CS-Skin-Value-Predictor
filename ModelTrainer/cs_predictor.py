import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.python.framework.dtypes as dtypes

import matplotlib.pyplot as plt

print(f"Numpy {np.__version__}")
print(f"Keras {keras.__version__}")
print(f"Tensorflow {tf.__version__}")

from keras import layers


def train_model():
    # load dataset
    (ds_train, ds_test) = tfds.as_numpy(
        tfds.load(name="cs_skin_price_data", split=["train", "test"], batch_size=-1, shuffle_files=True))

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

    # create layers

    """
    # do CNN on layers
    conv_1 = layers.Conv2D(filters=32, kernel_size=(8, 8), input_shape=(512, 512, 4))(texture_input)
    pool_1 = layers.MaxPooling2D((8, 8))(conv_1)
    conv_2 = layers.Conv2D(filters=32, kernel_size=(8, 8))(pool_1)
    pool_2 = layers.MaxPooling2D((4, 4))(conv_2)
    conv_3 = layers.Conv2D(filters=64, kernel_size=(6, 6))(pool_2)
    pool_3 = layers.MaxPooling2D((2, 2))(conv_3)
    conv_4 = layers.Conv2D(filters=128, kernel_size=(4, 4))(pool_3)
    """

    conv_1 = layers.Conv2D(filters=64, kernel_size=(16, 16))(texture_input)

    pool_1 = layers.MaxPooling2D((4, 4))(conv_1)

    conv_2 = layers.Conv2D(filters=16, kernel_size=(8, 8))(pool_1)
    pool_2 = layers.MaxPooling2D((14, 14))(conv_2)

    drop = layers.Dropout(rate=0.01)(pool_2)

    # flatten layer
    flattened = layers.Flatten()(drop)

    # compress layers
    dense_1 = layers.Dense(units=64, kernel_regularizer="l2")(flattened)

    dense_2 = layers.Dense(units=32, activation="elu")(dense_1)

    drop_4 = layers.Dropout(rate=0.1)(dense_2)

    # reshape data for concat
    weapon_input_reshaped = layers.Reshape((-1,))(weapon_input)
    rarity_input_reshaped = layers.Reshape((-1,))(rarity_input)

    # combine texture with weapon type and rarity
    combined = layers.Concatenate(axis=-1)([weapon_input_reshaped, rarity_input_reshaped, drop_4])

    # create output layer
    outputs = layers.Dense(units=1, activation="elu")(combined)

    # compile model
    model = keras.Model(inputs=[texture_input, weapon_input, rarity_input], outputs=outputs,
                        name="cs_predictor_model")
    model.summary()

    # MeanSquaredLogarithmicError

    model.compile(
        loss=keras.losses.MeanSquaredLogarithmicError(),
        optimizer=keras.optimizers.Nadam(),
        metrics=["accuracy"]
    )

    # train model
    history = model.fit(x_train, y_train, batch_size=32, epochs=16)

    # test model
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test loss: {test_scores[0]}")
    print(f"Test accuracy: {test_scores[1]}")

    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # save model
    model.save("cs_prediction.keras")


def predict():
    model = keras.models.load_model("cs_prediction.keras")

    img = keras.utils.img_to_array(
        keras.utils.load_img(input("File path: "), color_mode="rgba", target_size=(512, 512)))

    weapon_int = int(input("Weapon Int: "))
    rarity_int = int(input("Rarity Int: "))

    weapon_input = tf.constant([weapon_int])
    rarity_input = tf.constant([rarity_int])

    prediction = model([img, weapon_input, rarity_input])

    print(np.argmax(prediction))


def main():
    while True:
        # get user input
        x = input("Please select an option:\n1.) Train\n2.) Predict\n")

        if x == "1":
            train_model()
        elif x == "2":
            predict()


if __name__ == "__main__":
    # main()
    train_model()
