"""

CS2 Item Market Scraper

database_handler.py

Developed by Keagan Bowman
Copyright 2024

Uses data gathered from the source file reader to compile a list of skin names and tie them to their texture file counterparts

"""

import os
import json
import shutil
import sqlite3

from weapon_classifiers import get_weapon, get_rarity, WeaponToInt


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
            skin_tag_name TEXT,
            skin_weapon_type TEXT,
            skin_texture_file TEXT,
            skin_rarity INTEGER,
            skin_vmat_data TEXT DEFAULT "{}",
            skin_price_data TEXT DEFAULT "{}"
        );
        """)

        # re-add all VMATs to the DB
        for file in os.listdir("./VMATs"):

            # skip the missing_paintkit file
            if "missing_paintkit" in file:
                continue

            # get data_name from file name
            data_name = file.split(".")[0]

            # get file content
            with open(os.path.join("./VMATs", file), "r") as f:
                vmat_data = f.read()
                f.close()

            # add data to db
            cursor.execute("""
            INSERT INTO skins 
            (skin_data_name, skin_vmat_data)
            VALUES (?,?);
            """, (data_name, vmat_data))

        # close cursor
        cursor.close()

        # commit to db
        db.commit()

        # return db
        return db
    else:  # db exists and doesn't need to be wiped, so make a connection
        return sqlite3.connect(file_path)


def get_skin_data(items_json: dict, translations_json: dict, texture_folder: str, db: sqlite3.Connection) -> None:
    """
    Gathers texture data, tag names, rarities, and weapon types for all skins and adds it to the DB

    :param items_json: Dictionary object containing the contents of items_game
    :param translations_json: Dictionary object containing the contents of csgo_english
    :param texture_folder: The root folder where extracted textures are
    :param db: Database object to work on
    :return:
    """
    # item_sets

    skin_to_weapon = {}

    # gather collection list for weapon association
    for item_set in items_json["item_sets"].values():
        for item in item_set["items"].keys():
            # seperate the identifier from the weapon name
            item = item.split("]")

            # get identifier
            skin_identifier = item[0].lstrip("[")

            try:
                # get weapon name
                weapon_name = item[1]
            except IndexError:
                # invalid skin name, skip this collection
                continue

            if skin_identifier not in skin_to_weapon.keys():
                skin_to_weapon[skin_identifier] = get_weapon(weapon_name.split("weapon_")[-1])

    # populate skin names
    paint_kits = items_json["paint_kits"]
    for skin_id in paint_kits.keys():
        # gather skin data
        skin = paint_kits[skin_id]

        # skip default skins
        if skin["name"] == "default" or skin["name"] == "workshop_default":
            continue

        # get data name
        data_name = skin["name"]

        # get internal data tag name
        data_tag = skin["description_tag"].lstrip("#").lower()

        # skip "newcs2" items since they're duplicates
        if "newcs2" in data_tag:
            continue

        # get tag name
        tag_name = translations_json[data_tag]

        # update in db
        cursor = db.cursor()
        cursor.execute("UPDATE skins SET skin_tag_name = ? WHERE skin_data_name = ?;", (tag_name, data_name))

    # gather list of skins
    cursor = db.cursor()
    skin_data_names = db.execute("SELECT skin_data_name FROM skins;").fetchall()
    cursor.close()

    # populate texture, rarity, and weapon type information
    for data_name in skin_data_names:
        data_name = data_name[0]

        # check if skin already has texture in DB
        cursor = db.cursor()
        data = cursor.execute("SELECT skin_texture_file, skin_vmat_data FROM skins WHERE skin_data_name = ?",
                              (data_name,)).fetchone()
        cursor.close()

        # skip items that exist in the DB
        if data is None:
            continue
        elif data[0] is not None:
            continue

        # get vmat data
        vmat_data = json.loads(data[1])

        # get texture file from vmat data
        if vmat_data is not None and vmat_data != {}:
            # set chosen texture
            try:
                texture_name = vmat_data["Layer0"]["compiled textures"]["g_tpattern"].split("/")[-1]
            except KeyError:
                try:
                    texture_name = vmat_data["Layer0"]["compiled textures"]["g_tcolor"].split("/")[-1]
                except KeyError:
                    texture_name = None
        else:
            texture_name = None

        # replace .vtex with .png and ensure file exists
        if texture_name is not None:
            texture_name = texture_name.replace(".vtex",".png")

            if not os.path.exists(os.path.join(texture_folder, texture_name)):
                print(f"\t > Unable to find {texture_name}")

        # get skin rarity
        rarity = get_rarity(items_json['paint_kits_rarity'][data_name])

        # get weapon type
        try:
            weapon_type = skin_to_weapon[data_name].value
        except KeyError:
            weapon_type = WeaponToInt.KNIFE.value

        # create cursor
        cursor = db.cursor()

        # save to DB
        cursor.execute("""
                    UPDATE skins SET
                    skin_weapon_type = ?, skin_texture_file = ?, skin_rarity = ?
                    WHERE skin_data_name = ?;
                    """, (weapon_type, texture_name, rarity.value, data_name))

        # close cursor
        cursor.close()

        # commit to DB
        db.commit()
