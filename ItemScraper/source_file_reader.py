"""

CS2 Item Market Scraper

source_file_reader.py

Developed by Keagan Bowman
Copyright 2024

Reads Source data files into a dictionary

"""
import os
import re
import json
import subprocess


def gather_data(input_path: str, output_path: str, skip: int = 0) -> dict:
    """
    Collects data from the specified file path and returns it as a dictionary
    :param input_path: file path to parse
    :param output_path: file JSON to export to
    :param skip: number of lines to skip at the start of the file
    :return: file data as a dict object
    """

    # check if the output file exists
    if os.path.exists(output_path):
        # open file & gather data
        with open(output_path, "r") as f:
            file_json = json.load(f)
            f.close()

        print("\t > Loaded from preexisting file.")

        # return data
        return file_json
    else:
        # get data from file
        with open(input_path, "r", errors="ignore") as f:
            lines = f.readlines()
            f.close()

        for i in range(skip):
            lines.pop(0)

        line = lines.pop(0)

        # gather name of first key
        current_key_set = [line.replace("\n", "").replace("\"", "")]
        data = {current_key_set[0]: {}}

        # set current dictionary level
        current_level = data

        # create regex
        new_value = re.compile("\"(.*)\"\s*\"(.*)\"")
        new_value_multiline = re.compile("\"(.*)\"\s*\"(.*)")
        new_dict = re.compile("\"(.*)\"")

        include_next_line = False

        # handle every line in the file
        while len(lines) > 0:
            # get next line
            line = lines.pop(0)

            # strip tab spaces and normal from start and end of line
            line = line.lstrip("\t ").rstrip("\t ").replace("\n", "")

            # skip comment lines
            if line.startswith("//"):
                continue

            # add inline content
            if include_next_line:
                # add current line from list of keys with new line
                current_level[list(current_level.keys())[-1]] += "\n" + line.rstrip("\"")

                # check if this is the last line for the multi-line content
                if line.endswith("\""):
                    include_next_line = False

                continue

            # check for other cases
            if line == "":  # check for empty lines
                continue
            elif line == "{":  # move the dictionary level down one
                current_level = get_level(data, current_key_set)
            elif line == "}":  # move the dictionary level up one
                current_key_set.pop(-1)
                current_level = get_level(data, current_key_set)
            else:
                # get line values
                try:
                    # try to find new value
                    groups = new_value.match(line).groups()
                except AttributeError:
                    try:
                        # try to find multi-line values
                        groups = new_value_multiline.match(line).groups()

                        # mark that the next lines need to be included with this value
                        include_next_line = True
                    except AttributeError:
                        # try to find new dict
                        try:
                            # try to find new dictionary keys
                            groups = new_dict.match(line).groups()
                        except AttributeError as e:
                            print(line)
                            raise e

                if len(groups) == 1:  # new key
                    current_key_set.append(groups[0].lower())

                    # create new key dictionary if we don't have one already
                    if groups[0] not in current_level.keys():
                        current_level[groups[0].lower()] = {}
                else:  # add key/value pair to current dictionary level
                    current_level[groups[0].lower()] = groups[1]

        # return completed dictionary
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
            f.close()

        print("\t > Loaded from input file.")
        return data


def decompile_textures() -> None:
    """
    Uses the ValveResourceFormat decompiler to extract textures from a folder
    :return:
    """

    # get list of textures
    for root, folders, files in os.walk("../CSUnpacked/models/weapons/customization/paints/"):
        # check each file
        for file in files:
            # ensure this is a texture file
            if file.endswith(".vtex_c") and "normal" not in file:
                # get the file name
                output_name = file.split(".")[0] + ".png"

                # ensure file hasn't been decompiled
                if not os.path.exists(f"./ExportedTextures/{output_name}"):
                    # create command for the CLI
                    command = ["./TextureDecompiler/Decompiler.exe", "-i", os.path.join(root, file), "-o",
                               f"./ExportedTextures/{output_name}"]

                    # silently run command
                    subprocess.run(command, stdout=open(os.devnull, "wb"))


def get_level(dictionary: dict, arr: list) -> dict:
    """
    Travels down a root dictionary using a list of keys

    :param dictionary: Root dictionary node to check
    :param arr: List of keys to travel
    :return: Dictionary level navigated to with arr keys
    """

    current_level = dictionary
    for key in arr:
        current_level = current_level[key]
    return current_level
