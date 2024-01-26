"""

CS2 Item Market Scraper

dataset_builder.py

Developed by Keagan Bowman
Copyright 2024

Builds a dataset that is compatible with the tensorflow/keras model

"""

import sqlite3


def build_dataset(db: sqlite3.Connection) -> bool:
    """
    Builds a tensorflow/keras compatible dataset that can be used for the training of models using the data.

    :param db: Database that contains the data for the set
    :return: Returns a success boolean
    """

