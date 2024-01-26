"""

CS2 Item Market Scraper

main.py

Developed by Keagan Bowman
Copyright 2024

Gathers a list of all known skins and their marketplace name, texture file, and price data up to a given date

"""

from source_file_reader import gather_data, decompile_textures
from database_handler import build_db, match_textures
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

    print("Decompiling textures...")

    # decompile textures
    decompile_textures()

    print("Connecting to database...")

    db = build_db("./Data/skins.db", False)

    print("Matching textures...")

    match_textures(items_game_json, english_translations_json, "./ExportedTextures", db)

    print("Building dataset...")

    build_dataset(db)

    print("Process complete.")


main()
