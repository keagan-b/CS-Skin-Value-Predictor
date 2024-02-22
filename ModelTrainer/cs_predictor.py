import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import json
import keras
import sqlite3
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
    texture_input = keras.Input(shape=(512, 512, 4), name="texture_input")
    weapon_input = keras.Input(batch_shape=(1,), dtype=dtypes.int32, name="weapon_type_input")
    rarity_input = keras.Input(batch_shape=(1,), dtype=dtypes.int32, name="rarity_input")

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

    conv_1 = layers.Conv2D(filters=16, kernel_size=(16, 16), input_shape=(512, 512, 4))(texture_input)
    pool_1 = layers.MaxPooling2D((2, 2))(conv_1)
    # conv_2 = layers.Conv2D(filters=16, kernel_size=(4, 4))(pool_1)
    # pool_2 = layers.MaxPooling2D((4, 4))(conv_2)

    flat = layers.Flatten()(pool_1)

    norm = layers.Normalization()(flat)

    dense_1 = layers.Dense(units=64, activation="softplus")(norm)

    drop_1 = layers.Dropout(rate=0.05)(dense_1)

    dense_2 = layers.Dense(units=8, activation="leaky_relu")(drop_1)

    # reshape data for concat
    weapon_input_reshaped = layers.Reshape((-1,))(weapon_input)
    rarity_input_reshaped = layers.Reshape((-1,))(rarity_input)

    combined = layers.Concatenate(axis=-1)([weapon_input_reshaped, rarity_input_reshaped, dense_2])
    outputs = layers.Dense(units=1)(combined)

    # compile model
    model = keras.Model(inputs=[texture_input, weapon_input, rarity_input], outputs=outputs,
                        name="cs_predictor_model")
    model.summary()

    # MeanSquaredLogarithmicError

    # Adadelta is good :)

    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adadelta()
    )

    # train model
    history = model.fit(x_train, y_train, batch_size=64, epochs=64)

    # test model
    model.evaluate(x_test, y_test, verbose=2)

    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # save model
    model.save("cs_prediction.keras")


def predict_from_db(data_name: str, model):
    # connect to DB to get statistics
    db = sqlite3.connect("../ItemScraper/Data/skins.db")

    cursor = db.cursor()
    data = cursor.execute(
        "SELECT skin_price_data, skin_weapon_type, skin_rarity, skin_texture_file FROM skins WHERE skin_data_name = ?",
        (data_name,)).fetchone()
    cursor.close()

    weapon_int = int(data[1])
    rarity_int = int(data[2])
    file_path = data[3]

    price_data = json.loads(data[0])['prices']

    img = tf.convert_to_tensor(
        keras.utils.img_to_array(
            keras.utils.load_img(f"../ItemScraper/Textures/{file_path}", color_mode="rgba", target_size=(512, 512))
        ).reshape((-1, 512, 512, 4))
    )

    # weapon_int = int(input("Weapon Int: "))
    # rarity_int = int(input("Rarity Int: "))

    weapon_input = tf.constant([weapon_int])
    rarity_input = tf.constant([rarity_int])

    prediction = model([img, weapon_input, rarity_input])

    prices = []
    # gather all prices
    for price in price_data:
        # check prices
        # add parsed price
        prices.append(float(price[1]))

    prices = sorted(prices)

    print("Actual values:")

    # construct inter-quartile range
    quarter_size = len(prices) // 4
    q1 = prices[quarter_size * 1]
    q2 = prices[quarter_size * 2]
    q3 = prices[quarter_size * 3]
    iqr = (q3 - q1) * 1.5

    print(f"Q1: {q1}\nQ2: {q2}\nQ3: {q3}")
    print(f"Average: {sum(prices) / len(prices)}")

    # get min
    for price in prices:
        # ensure price is not an outlier
        if price < q1 - iqr:
            continue
        else:
            print(f"Min: {price}")
            break

    # get max
    for price in reversed(prices):
        # ensure price is not an outlier
        if price > q3 + iqr:
            continue
        else:
            print(f"Max: {price}")
            break

    print(f"Prediction: {prediction[0][0]}")


def predict_from_inputs(model):
    file_path = input("Texture file path: ")
    weapon_type = int(input("Weapon Type ID: "))
    rarity_type = int(input("Rarity ID: "))

    img = tf.convert_to_tensor(
        keras.utils.img_to_array(
            keras.utils.load_img(file_path, color_mode="rgba", target_size=(512, 512))
        ).reshape((-1, 512, 512, 4))
    )

    weapon_input = tf.constant([weapon_type])
    rarity_input = tf.constant([rarity_type])

    prediction = model([img, weapon_input, rarity_input])

    print(f"Prediction: {prediction[0][0]}")


def load_model():
    print("Loading model...")

    model = keras.models.load_model("cs_prediction.good.keras")

    print("Loaded model.")

    while True:
        pred_type = input("1.) From database\n2.) Custom file\n3.) Exit")

        if pred_type == "1":
            data_name = input("Please enter a skin data name\n")
            predict_from_db(data_name, model)
        elif pred_type == "2":
            predict_from_inputs(model)
        elif pred_type == "3":
            return


def main():
    while True:
        # get user input
        x = input("Please select an option:\n1.) Train\n2.) Predict\n3.) Both\n")

        if x == "1":
            train_model()
        elif x == "2":
            load_model()
        elif x == "3":
            train_model()
            load_model()


if __name__ == "__main__":
    main()
