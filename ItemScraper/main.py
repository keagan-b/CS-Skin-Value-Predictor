"""

CS2 Item Market Scraper

main.py

Developed by Keagan Bowman
Copyright 2024

Gathers a list of all known skins and their marketplace name, texture file, and price data up to a given date.
Will then build that into a dataset that's compatible with Keras.

"""

from source_file_reader import gather_data, decompile_textures, decompile_vmats
from database_handler import build_db, get_skin_data, get_prices
from dataset_builder import build_dataset


def main():
    """
    Primary function
    :return:
    """

    print("Collecting Item Game...")

    # gather data of all in game item data
    items_game_json = gather_data("Data/items_game.txt", "Data/items_game.json")["items_game"]

    print("Collecting Translations...")

    # gather all the english translation data
    english_translations_json = gather_data("Data/csgo_english.txt", "Data/csgo_english.json", 1)["lang"]["tokens"]

    print("Connecting to database...")

    db = build_db("./Data/skins.db", False)

    print("Decompiling VMATs...")

    # decompile VMATs
    decompile_vmats(db, "../CSUnpacked/models/weapons/customization/paints/vmats")

    print("Decompiling textures...")

    # decompile textures
    decompile_textures()

    print("Gathering skin information...")

    # get skin data for each skin with a valid VMAT file
    get_skin_data(items_game_json, english_translations_json, "./Textures", db)

    print("Gathering price data...")

    # send requests to steam API for price history
    get_prices(db)

    print("Building dataset...")

    # build dataset with TFDS
    build_dataset()

    print("Dataset built.")


main()
