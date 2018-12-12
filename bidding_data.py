import pandas as pd
import tensorflow as tf
import json
import config
import calculate_data_stats
import math

CONFIG = config.get_config()

REMOVE_COLUMNS = []
LABEL_COLUMN = None
DATA_STATS = None
STATS_FILES = None

def set_stats_file(stat_files):
    global STATS_FILES
    STATS_FILES = stat_files

def get_stats():
    global DATA_STATS
    if DATA_STATS is None:
        print("Calculating data statistics")
        calculate_data_stats.calculate_stats(STATS_FILES)
        with open(CONFIG['DATA_STATS_FILE'], 'r') as f:
            DATA_STATS = json.load(f)
        print("Finished calculating data statistics")

    return DATA_STATS

def get_column_names():
    return get_stats()['columns']['all']

def normalize(stats):
    fn = lambda x: (tf.to_float(x) - stats['mean']) / (stats['std'] + CONFIG["EPSILON"])
    return fn

def get_feature_columns():
    stats = get_stats()
    numeric_features = []
    for key in stats['columns']['numeric']:
        if key in get_remove_columns() + get_label_column():
            continue
        numeric_features.append(
            tf.feature_column.numeric_column(key, normalizer_fn = normalize(stats['stats'][key]))
        )

    categorical_features = []
    for key in stats['columns']['categorical']:
        if key in get_remove_columns() + get_label_column():
            continue
        stat = stats['stats'][key]
        embedding_size = math.ceil(stat['unique']**0.25)
        categorical_features.append(
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_hash_bucket(
                    key,
                    stat['unique'],
                    tf.string
                ),
                embedding_size + 1)
        )

    return numeric_features + categorical_features

def get_feature_columns_for_win_prediction():
    prepare_csv_column_list_for_win_prediction()
    return get_feature_columns()

def get_feature_columns_for_bid_prediction():
    prepare_csv_column_list_for_bid_prediction()
    return get_feature_columns()

def get_feature_columns_for_imp_prediction():
    prepare_csv_column_list_for_imp_prediction()
    return get_feature_columns()

def get_remove_columns():
    return REMOVE_COLUMNS

def set_remove_columns(remove_columns):
    global REMOVE_COLUMNS
    REMOVE_COLUMNS = remove_columns

def get_label_column():
    return LABEL_COLUMN

def set_label_column(label_column):
    global LABEL_COLUMN
    LABEL_COLUMN = label_column

""" Parse the CSV file of bidding data
Arguments:
    line: string, string of comma separated instance values
"""
def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, field_delim=CONFIG['CSV_SEPARATOR'], record_defaults=config.get_default_values_for_csv_columns(), na_value='null')

    # Pack the result into a dictionary
    features = dict(zip(get_column_names(), fields))

    for column in get_remove_columns():
        features.pop(column)

    # Separate the label from the features
    if get_label_column() is None:
        return features

    labels = []
    for label in get_label_column():
        labels.append(features.pop(label))

    return features, labels

def prepare_csv_column_list_for_win_prediction():
    set_remove_columns(CONFIG['PREDICT_WIN_REMOVE_FEATURES'])
    set_label_column(CONFIG['PREDICT_WIN_LABELS'])

def prepare_csv_column_list_for_bid_prediction():
    set_remove_columns(CONFIG['PREDICT_BID_REMOVE_FEATURES'])
    set_label_column(CONFIG['PREDICT_BID_LABELS'])

def prepare_csv_column_list_for_imp_prediction():
    set_remove_columns(CONFIG['PREDICT_IMP_REMOVE_FEATURES'])
    set_label_column(CONFIG['PREDICT_IMP_LABELS'])

""" WIN prediction """
def train_input_fn_for_predict_win(batch_size=1, num_epochs=1):
    filenames = CONFIG['PREDICT_WIN_DATASET_TRAIN']
    prepare_csv_column_list_for_win_prediction()

    return csv_input_fn(filenames, batch_size, num_epochs, is_shuffle=True)

def validation_input_fn_for_predict_win(batch_size=1, num_epochs=1):
    filenames = CONFIG['PREDICT_WIN_DATASET_VAL']
    prepare_csv_column_list_for_win_prediction()

    return csv_input_fn(filenames, batch_size, num_epochs, is_shuffle=True)

def test_input_fn_for_predict_win(filenames=None):
    if filenames is None:
        filenames = CONFIG['PREDICT_WIN_DATASET_TEST']
    prepare_csv_column_list_for_win_prediction()

    return csv_input_fn(filenames, batch_size=1, num_epochs=1, is_shuffle=False)

""" BID prediction """
def train_input_fn_for_predict_bid(batch_size=1, num_epochs=1):
    filenames = CONFIG['PREDICT_BID_DATASET_TRAIN']
    prepare_csv_column_list_for_bid_prediction()

    return csv_input_fn(filenames, batch_size, num_epochs, is_shuffle=True)

def validation_input_fn_for_predict_bid(batch_size=1, num_epochs=1):
    filenames = CONFIG['PREDICT_BID_DATASET_VAL']
    prepare_csv_column_list_for_bid_prediction()

    return csv_input_fn(filenames, batch_size, num_epochs, is_shuffle=True)

def test_input_fn_for_predict_bid(filenames=None):
    if filenames is None:
        filenames = CONFIG['PREDICT_BID_DATASET_TEST']
    prepare_csv_column_list_for_bid_prediction()

    return csv_input_fn(filenames, batch_size=1, num_epochs=1, is_shuffle=False)

""" IMP prediction """
def train_input_fn_for_predict_imp(batch_size=1, num_epochs=1):
    filenames = CONFIG['PREDICT_IMP_DATASET_TRAIN']
    prepare_csv_column_list_for_imp_prediction()

    return csv_input_fn(filenames, batch_size, num_epochs, is_shuffle=True)

def validation_input_fn_for_predict_imp(batch_size=1, num_epochs=1):
    filenames = CONFIG['PREDICT_IMP_DATASET_VAL']
    prepare_csv_column_list_for_imp_prediction()

    return csv_input_fn(filenames, batch_size, num_epochs, is_shuffle=True)

def test_input_fn_for_predict_imp(filenames=None):
    if filenames is None:
        filenames = CONFIG['PREDICT_IMP_DATASET_TEST']
    prepare_csv_column_list_for_imp_prediction()

    return csv_input_fn(filenames, batch_size=1, num_epochs=1, is_shuffle=False)

""" Return dataset in batches from a CSV file
Arguments:
    csv_path: string, CSV path file
    batch_size: integer, Number of instances to return
Returns:
    dataset tensor parsed from csv
"""
def csv_input_fn(filenames, batch_size, num_epochs, is_shuffle=True):

    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    dataset = dataset.flat_map(lambda filename: tf.data.TextLineDataset(filename).skip(1))

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    if is_shuffle:
        dataset = dataset.shuffle(10 * batch_size, seed=42).repeat(count=None)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

    # Return the dataset.
    # return dataset
