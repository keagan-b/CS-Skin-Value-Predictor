"""

CS2 Item Market Scraper

weapon_classifier.py

Developed by Keagan Bowman
Copyright 2024

Enum declarations for classifying Counter Strike weapons to string/int counterparts

"""

import re
import enum


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
    M4A4 = 24
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


str_to_weapon = {
    "p2000": WeaponToInt.P2000,
    "usp-s": WeaponToInt.USPS,
    "usps": WeaponToInt.USPS,
    "glock-18": WeaponToInt.GLOCK,
    "glock18": WeaponToInt.GLOCK,
    "p250": WeaponToInt.P250,
    "five-seven": WeaponToInt.FIVESEVEN,
    "fiveseven": WeaponToInt.FIVESEVEN,
    "tec-9": WeaponToInt.TEC9,
    "tec9": WeaponToInt.TEC9,
    "cz-75": WeaponToInt.CZ75,
    "cz75": WeaponToInt.CZ75,
    "dual berettas": WeaponToInt.DUALBERETTAS,
    "desert eagle": WeaponToInt.DEAGLE,
    "deagle": WeaponToInt.DEAGLE,
    "r8 revolver": WeaponToInt.R8,
    "revolver": WeaponToInt.R8,
    "nova": WeaponToInt.NOVA,
    "xm1014": WeaponToInt.XM1014,
    "mag-7": WeaponToInt.MAG7,
    "mag7": WeaponToInt.MAG7,
    "sawed-off": WeaponToInt.SAWEDOFF,
    "sawedoff": WeaponToInt.SAWEDOFF,
    "mp-9": WeaponToInt.MP9,
    "mp9": WeaponToInt.MP9,
    "mac-10": WeaponToInt.MAC10,
    "mac10": WeaponToInt.MAC10,
    "pp-bizon": WeaponToInt.PPBIZON,
    "ppbizon": WeaponToInt.PPBIZON,
    "mp7": WeaponToInt.MP7,
    "ump-45": WeaponToInt.UMP45,
    "ump45": WeaponToInt.UMP45,
    "p90": WeaponToInt.P90,
    "mp5": WeaponToInt.MP5,
    "famas": WeaponToInt.FAMAS,
    "galil ar": WeaponToInt.GALIL,
    "m4a4": WeaponToInt.M4A4,
    "m4a1-s": WeaponToInt.M4A1S,
    "ak-47": WeaponToInt.AK47,
    "ak47": WeaponToInt.AK47,
    "ak": WeaponToInt.AK47,
    "aug": WeaponToInt.AUG,
    "sg 553": WeaponToInt.SG553,
    "ssg 08": WeaponToInt.SSG08,
    "awp": WeaponToInt.AWP,
    "scar-20": WeaponToInt.SCAR20,
    "g3sg1": WeaponToInt.G3SG1,
    "m249": WeaponToInt.M249,
    "negev": WeaponToInt.NEGEV,
}


def classify_weapon(input_str: str) -> WeaponToInt:
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
