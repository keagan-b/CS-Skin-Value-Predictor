"""

CS2 Item Market Scraper

dataset_builder.py

Developed by Keagan Bowman
Copyright 2024

Builds a dataset that is compatible with the tensorflow/keras model

"""

import os
import subprocess


def build_dataset():
    """
    Builds a tensorflow/keras compatible dataset that can be used for the training of models using the data.
    """

    # run build command
    subprocess.run(cwd=".\\cs_skin_price_data", args=["tfds", "build", "--overwrite"], stdout=open(os.devnull, "wb"), stderr=open(os.devnull, "wb"))
