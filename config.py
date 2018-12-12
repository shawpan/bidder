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
#2|12440|8084|23678|null|null|300x250|0.707278907|null|15166603496|4|2|19930|100000|null|null|0.315631807|null|2152980677|84.241.195.0|75|Amsterdam|4|0.856686056|null|0|0|0.8566860556602478
def get_types_of_attributes():
    return {
        'deliveryid' : np.string_,          # 307219911482
        'dayofweek' : np.int_,              # 3
        'hour' : np.int_,                   # 01
        'pub_sspid' : np.string_,           # 2
        'pub_accountid' : np.string_,       # 12440
        'pub_as_siteId' : np.string_,       # 8084
        'pub_as_adspaceid' : np.string_,    # 23678
        'pub_as_domain' : np.string_,       # null
        'pub_as_pageurl' : np.string_,      # null
        'pub_as_dimensions' : np.string_,   # 300x250
        'pub_as_viewrate' : np.double,      # 0.707278907
        'pub_as_position' : np.string_,     # null
        'pub_as_caps' : np.string_,         # 15166603496
        'req_buymodel' : np.string_,        # 4
        'req_auctiontype' : np.string_,     # 2
        'device_os' : np.string_,           # 19930
        'device_model' : np.string_,        # 100000
        'rtb_ctr' : np.double,              # null
        'rtb_viewrate' : np.double,         # null
        'rtb_bidfloor' : np.double,         # 0.315631807
        'rtb_battr' : np.string_,           # null
        'rtb_tagId' : np.string_,           # 2152980677
        'user_ip' : np.string_,             # 84.241.195.0
        'user_market' : np.string_,         # 75
        'user_city' : np.string_,           # Amsterdam
        'ad_imptype' : np.string_,          # 4
        'req_bid' : np.double,              # 0.856686056
        'price' : np.double,                # null
        'won' : np.double,                  # 0
        'imp' : np.int_,                    # 0
        'targetBid' : np.double,            # 0.8566860556602478
    }

def get_default_values_for_csv_columns():
    default_value_for_dtypes = {
        np.string_: "",
        np.int_: 0,
        np.double: 0.0
    }
    types_of_attributes = get_types_of_attributes()

    return [ default_value_for_dtypes[dtype] for column_name, dtype in types_of_attributes.items() ]

# print(get_default_values_for_csv_columns())
