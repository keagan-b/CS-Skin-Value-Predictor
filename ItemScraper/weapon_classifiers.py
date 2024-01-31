"""

CS2 Item Market Scraper

weapon_classifier.py

Developed by Keagan Bowman
Copyright 2024

Enum declarations for classifying Counter Strike weapons to string/int counterparts

"""

import re
import enum
from typing import Any


class WeaponToInt(enum.Enum):
    P2000 = 1
    USPS = 2
    GLOCK = 3
    P250 = 4
    FIVESEVEN = 5
    TEC9 = 6
    CZ75 = 7
    DUALBERETTAS = 8
    DEAGLE = 9
    R8 = 10
    NOVA = 11
    XM1014 = 12
    MAG7 = 13
    SAWEDOFF = 14
    MP9 = 15
    MAC10 = 16
    PPBIZON = 17
    MP7 = 18
    UMP45 = 19
    P90 = 20
    MP5 = 21
    FAMAS = 22
    GALIL = 23
    M4A1 = 24
    M4A1S = 25
    AK47 = 26
    AUG = 27
    SG553 = 28
    SSG08 = 29
    AWP = 30
    SCAR20 = 31
    G3SG1 = 32
    M249 = 33
    NEGEV = 34
    KNIFE = 35


class RarityToInt(enum.Enum):
    Common = 0
    Uncommon = 1
    Rare = 2
    Mythical = 3
    Legendary = 4
    Ancient = 5
    Immortal = 6


str_to_rarity = {
    "common": RarityToInt.Common,
    "uncommon": RarityToInt.Uncommon,
    "rare": RarityToInt.Rare,
    "mythical": RarityToInt.Mythical,
    "legendary": RarityToInt.Legendary,
    "ancient": RarityToInt.Ancient,
    "immortal": RarityToInt.Immortal
}

str_to_weapon = {
    "tec9": WeaponToInt.TEC9,
    "ssg08": WeaponToInt.SSG08,
    "elite": WeaponToInt.DUALBERETTAS,
    "galilar": WeaponToInt.GALIL,
    "p90": WeaponToInt.P90,
    "cz75a": WeaponToInt.CZ75,
    "hkp2000": WeaponToInt.P2000,
    "aug": WeaponToInt.AUG,
    "bizon": WeaponToInt.PPBIZON,
    "mac10": WeaponToInt.MAC10,
    "xm1014": WeaponToInt.XM1014,
    "m4a1_silencer": WeaponToInt.M4A1S,
    "scar20": WeaponToInt.SCAR20,
    "usp_silencer": WeaponToInt.USPS,
    "ak47": WeaponToInt.AK47,
    "m4a1": WeaponToInt.M4A1,
    "mp7": WeaponToInt.MP7,
    "sg556": WeaponToInt.SG553,
    "glock": WeaponToInt.GLOCK,
    "deagle": WeaponToInt.DEAGLE,
    "awp": WeaponToInt.AWP,
    "mag7": WeaponToInt.MAG7,
    "famas": WeaponToInt.FAMAS,
    "sawedoff": WeaponToInt.SAWEDOFF,
    "p250": WeaponToInt.P250,
    "nova": WeaponToInt.NOVA,
    "ump45": WeaponToInt.UMP45,
    "g3sg1": WeaponToInt.G3SG1,
    "fiveseven": WeaponToInt.FIVESEVEN,
    "mp9": WeaponToInt.MP9,
    "m249": WeaponToInt.M249,
    "negev": WeaponToInt.NEGEV,
    "revolver": WeaponToInt.R8,
    "mp5sd": WeaponToInt.MP5,
}


def classify_weapon(input_str: str) -> WeaponToInt | None:
    """
    Attempts to classify a weapon given a string
    :param input_str: string with potential weapon name in it
    :return: returns either a WeaponToInt value or None if no weapon was found
    """

    # compile regex
    splits = re.compile("[\s_]")

    # split string on spaces and underscores
    data = splits.split(input_str)

    # set return value to None
    weapon = None

    for part in data:
        part = part.lower()

        try:
            # set weapon value
            weapon = str_to_weapon[part]

            # break from loop
            break
        except KeyError:
            # ignore key errors
            continue

    # return weapon value
    return weapon


def get_weapon(input_str: str) -> WeaponToInt | None:
    """
    Returns a WeaponToInt value based on the input string.

    :param input_str: string to attmept to translate
    :return: weapon value, None if no valid value is found
    """

    input_str = input_str.lower()
    if input_str in str_to_weapon.keys():
        return str_to_weapon[input_str]
    else:
        return None


def get_rarity(input_str: str) -> RarityToInt | None:
    """
    Returns a RarityToInt value based on the input string.

    :param input_str: string to attmept to translate
    :return: rarity value, None if no valid value is found
    """

    input_str = input_str.lower()
    if input_str in str_to_rarity.keys():
        return str_to_rarity[input_str]
    else:
        return None
