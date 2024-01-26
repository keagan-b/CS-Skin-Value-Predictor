"""

CS2 Item Market Scraper

database_handler.py

Developed by Keagan Bowman
Copyright 2024

Uses data gathered from the source file reader to compile a list of skin names and tie them to their texture file counterparts

"""

import os
import shutil
import sqlite3

from weapon_classifier import classify_weapon


def build_db(file_path: str, should_wipe: bool = False) -> sqlite3.Connection:
    """
    Handles the creation and wiping of the database
    :param file_path: Location of the database
    :param should_wipe: Should the database be wiped? Will require user input if set to True.
    :return: Returns a connection to the database
    """

    # check conditions to build/wipe DB
    if not os.path.exists(file_path) or should_wipe:

        # confirm database wipe
        if should_wipe:
            input("Please press [enter] to confirm the database wipe.\n")

        # connect to DB
        db = sqlite3.connect(file_path)

        # create cursor
        cursor = db.cursor()

        # drop table
        cursor.execute("DROP TABLE IF EXISTS skins;")

        # generate table
        cursor.execute("""
        CREATE TABLE skins (
            skin_data_name TEXT PRIMARY KEY,
            skin_tag_name TEXT NOT NULL,
            skin_weapon_type TEXT NOT NULL,
            skin_texture_file TEXT,
            skin_rarity INTEGER,
            skin_price_data TEXT DEFAULT "{}"
        );
        """)

        # close cursor
        cursor.close()

        # commit to db
        db.commit()

        # return db
        return db
    else:  # db exists and doesn't need to be wiped, so make a connection
        return sqlite3.connect(file_path)


def match_textures(items_json: dict, translations_json: dict, texture_folder: str, db: sqlite3.Connection) -> None:
    """
    Digs through the translation and item JSON to find and match textures to skins.
    :param items_json: Dictionary object containing the contents of items_game
    :param translations_json: Dictionary object containing the contents of csgo_english
    :param texture_folder: The root folder where extracted textures are
    :param db: Database object to work on
    :return:
    """

    # collect all texture files
    files = os.listdir("./ExportedTextures")

    # create a list of all texture file names
    file_names = [file for file in files]

    # gather list of skins
    paint_kits = items_json["paint_kits"]
    for skin_id in paint_kits.keys():
        # gather skin data
        skin = paint_kits[skin_id]

        # skip default skins
        if skin["name"] == "default" or skin["name"] == "workshop_default":
            continue

        # get data name
        data_name = skin["name"]

        # check if tag is already in DB
        cursor = db.cursor()
        data = cursor.execute("SELECT skin_texture_file FROM skins WHERE skin_data_name = ?", (data_name, )).fetchone()
        cursor.close()

        # skip items that exist in the DB and remove their textures
        if data is not None and len(data) > 0:
            for texture in data:
                try: file_names.remove(texture)
                except ValueError: pass
            continue

        # get internal data tag name
        data_tag = skin["description_tag"].lstrip("#").lower()

        # skip "newcs2" items since they're duplicates
        if "newcs2" in data_tag:
            continue

        # get tag name
        tag_name = translations_json[data_tag]

        # create valid textures array
        valid = []

        # get all parts of the tag
        for part in data_name.lower().split("_"):
            # make sure part is not weapon name and part len > 2
            if classify_weapon(part) is None and len(part) > 2:
                # loop through all file names
                for path in file_names:
                    # check to see if parts match
                    if part in path:
                        # check if path is already valid
                        if path not in valid:
                            # add path to valid
                            valid.append(path)

                            # make sure the temp folder exists
                            if not os.path.exists("./temp"):
                                os.mkdir("./temp")

                            # copy texture file over
                            shutil.copy(f"./ExportedTextures/{path}", f"./temp/{len(valid)}.png")
                            break

        # set chosen texture
        chosen_texture = None

        # check valid length
        match len(valid):
            case 0:
                # invalid length
                print(f"\t > No valid textures for {tag_name} ({data_name}).")
            case 1:
                # valid length, remove from valid
                chosen_texture = valid.pop(0)
            case _:
                # have user choose one of the valid textures
                print(f"\t > More than one valid texture found for {tag_name} ({data_name}).")

                # create a string listing out all the valid texture files
                textures_string = "\n\t\t > ".join(f"{x + 1} {valid[x]}" for x in range(len(valid)))

                # error-handling loop - ensures a chosen answer is valid
                while True:
                    try:
                        # get chosen texture
                        chosen_texture = int(
                            input(f"\t > Please choose a texture (-1 for None):\n\t\t > {textures_string}\n\t\t > "))

                        # ensure texture hasn't been set to None
                        if chosen_texture == -1:
                            chosen_texture = None
                        else:
                            chosen_texture = valid.pop(chosen_texture - 1)
                    except IndexError:
                        continue

                    # wipe valid list
                    valid = []
                    break

        # remove temp folder
        if os.path.exists("./temp"):
            shutil.rmtree("./temp")

        # make sure there is a chosen texture
        if chosen_texture:
            # remove this texture from the list
            file_names.remove(chosen_texture)

            # create cursor
            cursor = db.cursor()

            # save to DB
            cursor.execute("""
                INSERT INTO skins
                (skin_data_name, skin_tag_name, skin_weapon_type, skin_texture_file)
                VALUES (?,?,?,?);
                """, (data_name, tag_name, 0, chosen_texture))

            # close cursor
            cursor.close()

            # commit to DB
            db.commit()


def gather_rarities() -> None:
    pass