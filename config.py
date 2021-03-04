import os
import configparser
from typing import Any, NoReturn

# Config object
config = configparser.ConfigParser()

# D:\Data\EyesSimulation Sessions\Export3

def init_config(fn: str=".\\set_locations.ini") -> NoReturn:
    """
    Initializes the config object.
     :param fn: path of file to save to or read from.
     :return: -
    """
    if not os.path.exists(fn):
        create_config(fn)
    config.read(fn)


def create_config(fn: str) -> NoReturn:
    """
    Create a config file
    :param fn: path of file to save to.
    :return: -
    """
    config.add_section("DataPaths")
    config.set("DataPaths", "train_data",
               "C:\\Users\\airen\\Projects\\EyeGazeTesting\\Verification Task\\GazeVerificationStand\\data\\train")
    config.set("DataPaths", "owner_data",
               "C:\\Users\\airen\\Projects\\EyeGazeTesting\\Verification Task\\GazeVerificationStand\\data\\owner")
    config.set("DataPaths", "run_data",
               "C:\\Users\\airen\\Projects\\EyeGazeTesting\\Verification Task\\GazeVerificationStand\\data\\run")
    config.set("DataPaths", "selected_columns",
               "C:\\Users\\airen\\Projects\\EyeGazeTesting\\Verification Task\\GazeVerificationStand\\settings\\selected_columns.csv")

    config.add_section("EyemovementClassification")
    config.set("EyemovementClassification", "filtering_params",
               "C:\\Users\\airen\\Projects\\EyeGazeTesting\\Verification Task\\GazeVerificationStand\\settings\\filtering_params.json")
    config.set("EyemovementClassification", "model_params",
               "C:\\Users\\airen\\Projects\\EyeGazeTesting\\Verification Task\\GazeVerificationStand\\settings\\ivdt_params.json")

    config.add_section("FeatureGeneration")
    config.set("FeatureGeneration", "processing_params",
               "C:\\Users\\airen\\Projects\\EyeGazeTesting\\Verification Task\\GazeVerificationStand\\settings\\processing_params.json")
    config.set("FeatureGeneration", "features_params",
               "C:\\Users\\airen\\Projects\\EyeGazeTesting\\Verification Task\\GazeVerificationStand\\settings\\features_params.json")

    config.add_section("GazeVerification")
    config.set("GazeVerification", "model_params",
               "C:\\Users\\airen\\Projects\\EyeGazeTesting\\Verification Task\\GazeVerificationStand\\settings\\model_params.json")
    config.set("GazeVerification", "pretrained_model_location",
               "C:\\Users\\airen\\Projects\\EyeGazeTesting\\Verification Task\\GazeVerificationStand\\models_checkpoints")
    config.set("GazeVerification", "pretrained_model_fn",
               "C:\\Users\\airen\\Projects\\EyeGazeTesting\\Verification Task\\GazeVerificationStand\\models_checkpoints\\model_test2.pt")
    config.set("GazeVerification", "verification_params",
               "C:\\Users\\airen\\Projects\\EyeGazeTesting\\Verification Task\\GazeVerificationStand\\settings\\verification_params.json")

    with open(fn, "w") as config_file:
        config.write(config_file)



def get_setting(section: str, setting: str) -> Any:
    """
    Print out a setting.
    :param section: Section of config
    :param setting: Desired setting
    :return: settings value
    """
    value = config.get(section, setting)
    print(f"Setting from: {section} {setting} is - {value}")
    return value


def update_setting(fn: str, section: str,
                   setting: str, value: Any) -> NoReturn:
    """
    Update a setting in file.
    :param section: Section of config
    :param setting: Desired setting
    :param value: Value to set
    :return: -
    """
    config.set(section, setting, value)
    with open(fn, "w") as config_file:
        config.write(config_file)


def delete_setting(fn: str, section: str,
                   setting: str) -> NoReturn:
    """
    Delete a setting in file
    :param section: Section of config
    :param setting: Desired setting
    :return: -
    """
    config.remove_option(section, setting)
    with open(fn, "w") as config_file:
        config.write(config_file)


if __name__ == "__main__":
    path = ".\\set_locations.ini"
    init_config(path)