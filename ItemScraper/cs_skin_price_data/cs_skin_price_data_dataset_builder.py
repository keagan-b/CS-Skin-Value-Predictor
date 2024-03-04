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
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'texture': tfds.features.Image(shape=(512, 512, 4)),
                'price': tfds.features.Scalar(dtype=dtypes.float32),
                'rarity': tfds.features.Scalar(dtype=dtypes.int16),
                'weapon_type': tfds.features.Scalar(dtype=dtypes.int16)
            })
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
            price_data = json.loads(skin[4])["prices"]

            prices = []

            # gather all prices
            for price in price_data:
                # check prices
                # if "Jan" in price[0] and "2024" in price[0]:
                # add parsed price
                prices.append(float(price[1]))

            prices = sorted(prices)

            # construct inter-quartile range
            quarter_size = len(prices) // 4
            q1 = prices[quarter_size * 1]
            q2 = prices[quarter_size * 2]
            q3 = prices[quarter_size * 3]
            iqr = (q3 - q1) * 1.5

            # add IQR to price data
            final_prices = [
                ("q1", q1),
                ("q2", q2),
                ("q3", q3),
                ("avg", sum(prices) / len(prices))
            ]

            # get min
            for price in prices:
                # ensure price is not an outlier
                if price < q1 - iqr:
                    continue
                else:
                    # final_prices.append(("min", price))
                    break

            # get max
            for price in reversed(prices):
                # ensure price is not an outlier
                if price > q3 + iqr:
                    continue
                else:
                    # final_prices.append(("max", price))
                    break

            # construct dataset
            for tag_name, value in final_prices:
                # return example with skin_data_name as key
                yield f"{skin[0]}_{tag_name}", {
                    'texture': f"../Textures/{skin[2]}",
                    'price': value,
                    'rarity': int(skin[3]),
                    'weapon_type': int(skin[1])
                }
