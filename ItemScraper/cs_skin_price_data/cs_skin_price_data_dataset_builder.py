"""cs_skin_price_data dataset."""

import sqlite3
import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for cs_skin_price_data dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(cs_skin_price_data): Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'texture': tfds.features.Image(shape=(None, None, 3)),
                'price': tfds.features.Scalar(dtype="float32"),
                'rarity': tfds.features.Scalar(dtype="int32"),
                'weapon_type': tfds.features.Scalar(dtype="int32")
            })
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(cs_skin_price_data): Downloads the data and defines the splits
        path = dl_manager.download_and_extract('https://todo-data-url')

        # TODO(cs_skin_price_data): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train': self._generate_examples(path / 'train_imgs'),
        }

    def _generate_examples(self, database_path):
        # gathers data from the DB
        db = sqlite3.connect(database_path)

        # creates cursor and gathers data
        cursor = db.cursor()
        data = cursor.execute("SELECT * FROM skins").fetchall()
        cursor.close()

        """Yields examples."""
        # TODO(cs_skin_price_data): Yields (key, example) tuples from the dataset
        for skin in data:
            yield skin[0], {
                'weapon_type': skin[2],
                'texture': skin[3],
                'price': skin[5],
                'rarity': skin[4]
            }
