import pandas as pd
import numpy as np
import config
import json

CONFIG = config.get_config()
all_files = CONFIG['PREDICT_BID_DATASET_TRAIN']
dtypes = config.get_types_of_attributes()

def calculate_stats():
    df = pd.concat((pd.read_csv(f, sep=CONFIG['CSV_SEPARATOR'], na_values=["null"], dtype=dtypes) for f in all_files))
    stats_categorical = json.loads(df.describe(include='O').loc[[
        'count', 'unique'
    ]].to_json())
    stats_numeric = json.loads(df.describe().loc[[
        'count', 'mean', 'std', 'min', 'max'
    ]].to_json())
    columns = df.columns.values
    with open(CONFIG['DATA_STATS_FILE'], "w") as f:
        json.dump(obj={
            'columns': {
                'all': columns.tolist(),
                'categorical': list(stats_categorical.keys()),
                'numeric': list(stats_numeric.keys())
            },
            'stats': { **stats_numeric , **stats_categorical }
        }, fp=f, indent=4)

calculate_stats()
