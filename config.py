import json
import numpy as np

CONFIG = None

"""
Get the configurations from config.json file as object
"""
def get_config():
    global CONFIG
    if CONFIG is None:
        with open('config.json', 'r') as f:
            CONFIG = json.load(f)

    return CONFIG

def get_types_of_attributes():
    return {
        'sspid' : np.string_,
        'accountid' : np.string_,
        'device_os' : np.string_,
        'device_model' : np.string_,
        'market' : np.string_,
        'businessmodelid' : np.string_
    }
