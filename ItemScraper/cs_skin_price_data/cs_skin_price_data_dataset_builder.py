"""cs_skin_price_data dataset."""

import os
import json
import math
import sqlite3
import tensorflow_datasets as tfds
import tensorflow.python.framework.dtypes as dtypes


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for cs_skin_price_data dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Skin data must be collected and placed in the skins.db file, located in ../Data/skins.db
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(cs_skin_price_data): Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'texture': tfds.features.Image(shape=(1024, 1024, 4)),
                'price': tfds.features.Scalar(dtype=dtypes.float32),
                'rarity': tfds.features.Scalar(dtype=dtypes.int32),
                'weapon_type': tfds.features.Scalar(dtype=dtypes.int32)
            })
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        # gather skin data from DB
        db = sqlite3.connect('../Data/skins.db')
        cursor = db.cursor()
        data = db.execute(
            "SELECT skin_data_name, skin_weapon_type, skin_texture_file, skin_rarity, skin_price_data FROM skins WHERE skin_weapon_type != 35 AND skin_price_data != \"{}\"").fetchall()
        cursor.close()

        checked_data = []

        # remove any skins that have invalid texture paths
        for skin in data:
            if os.path.exists(f"../Textures/{skin[2]}"):
                checked_data.append(skin)

        # determine the size of training vs test data
        train_size = math.floor(len(checked_data) * 0.8)

        # create train vs test data lists
        train_data = checked_data[:train_size]
        test_data = checked_data[train_size:]

        # generate examples
        return {
            'train': self._generate_examples(train_data),
            'test': self._generate_examples(test_data)
        }

    def _generate_examples(self, data):

        """Yields examples."""
        for skin in data:
            # parse JSON data
            prices = json.loads(skin[4])["prices"]

            # get price of skin of Feb 1st, 2024, 12:00 AM
            feb_1st_price = 0
            for price in prices:
                # check prices
                if price[0] == "Feb 01 2024 00: +0":
                    # set parsed price
                    feb_1st_price = price[1]
                    break

            # return example with skin_data_name as key
            yield skin[0], {
                'texture': f"../Textures/{skin[2]}",
                'price': float(feb_1st_price),
                'rarity': int(skin[3]),
                'weapon_type': int(skin[1])
            }
