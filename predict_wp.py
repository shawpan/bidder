""" Train, Evaluate and Predict Winning Rate of a Bidding Request """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
from tensorflow.python.ops import math_ops
# import tensorflow_probability as tfp
# tfd = tfp.distributions
import shutil
import functools
import adanet
from adanet.examples import simple_dnn

import bidding_data
import config
import pandas as pd
import numpy as np

CONFIG = config.get_config()
OUTPUT_DIR = CONFIG['OUTPUT_DIR_PREDICT_WR']

BATCH_SIZE = 512
NUM_EPOCHS = 4000
EVAL_STEPS = 100

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='batch size')
parser.add_argument('--clean', default=0, type=int, help='Clean previously trained data')
parser.add_argument('--is_test', default=0, type=int, help='Is Test')
parser.add_argument('--train_steps', default=NUM_EPOCHS, type=int,
                    help='number of training steps')

def get_model():
    return get_adanet_model()

def ensemble_architecture(result):
  """Extracts the ensemble architecture from evaluation results."""

  architecture = result["architecture/adanet/ensembles"]
  # The architecture is a serialized Summary proto for TensorBoard.
  summary_proto = tf.summary.Summary.FromString(architecture)
  return summary_proto.value[0].tensor.string_val[0]

def first_price_auction_loss(labels, logits, features):
    labels = math_ops.to_float(labels)
    logits = math_ops.to_float(logits)
    wons = tf.reshape(features['won'], [-1,1])
    cdf = tf.distributions.Normal(loc=0.0, scale=1.0).cdf(tf.math.l2_normalize(labels - logits))
    loss_on_won = tf.math.multiply((1. - wons), tf.log(cdf))
    loss_on_losts = tf.math.multiply(wons, (tf.log(1.0 - cdf)))
    loss = tf.math.multiply(-1.0, tf.math.add(loss_on_won, loss_on_losts))

    return loss

def get_adanet_model():
    LEARNING_RATE = 0.003  #@param {type:"number"}
    TRAIN_STEPS = NUM_EPOCHS  #@param {type:"integer"}
    # BATCH_SIZE = 64  #@param {type:"integer"}
    ADANET_ITERATIONS = 8  #@param {type:"integer"}

    RANDOM_SEED = 42
    # Estimator configuration.
    runConfig = tf.estimator.RunConfig(
        save_checkpoints_steps=100,
        save_summary_steps=100,
        tf_random_seed=RANDOM_SEED)
    classifier = estimator = adanet.Estimator(
        model_dir = OUTPUT_DIR,
        # adanet_loss_decay=0.99,
        head=tf.contrib.estimator.regression_head(
        label_dimension=1,
        loss_fn=first_price_auction_loss
        ),
        subnetwork_generator=simple_dnn.Generator(
            learn_mixture_weights=True,
            dropout=0.5,
            feature_columns=bidding_data.get_feature_columns_for_wp_prediction(),
            optimizer=tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE),
            seed=RANDOM_SEED),
        max_iteration_steps=TRAIN_STEPS // ADANET_ITERATIONS,
        evaluator=adanet.Evaluator(
            input_fn=lambda : bidding_data.validation_input_fn_for_predict_wp(batch_size=BATCH_SIZE, num_epochs=TRAIN_STEPS),
            steps=EVAL_STEPS),
        config=runConfig)

    return classifier

""" Train the model """
def train_and_evaluate():
    estimator = get_model()
    serving_feature_spec = tf.feature_column.make_parse_example_spec(
      bidding_data.get_feature_columns_for_wp_prediction())
    serving_input_receiver_fn = (tf.estimator.export.build_parsing_serving_input_receiver_fn(serving_feature_spec))

    exporter = tf.estimator.BestExporter(
      name="best_exporter",
      serving_input_receiver_fn=serving_input_receiver_fn,
      exports_to_keep=5)
    train_spec = tf.estimator.TrainSpec(
                       input_fn = lambda : bidding_data.train_input_fn_for_predict_wp(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS),
                       max_steps = NUM_EPOCHS)
    eval_spec = tf.estimator.EvalSpec(
                       input_fn = lambda : bidding_data.validation_input_fn_for_predict_wp(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS),
                       steps = EVAL_STEPS,
                       # exporters=exporter,
                       start_delay_secs = 1, # start evaluating after N seconds
                       throttle_secs = 10)
    results, _  = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    if results is not None:
        print(results)
        print("Architecture:", ensemble_architecture(results))

def test_with_print(estimator, filenames, set_name=''):
    dtypes = config.get_types_of_attributes()
    df = pd.concat((pd.read_csv(f, na_values=["(null)"], dtype=dtypes) for f in filenames))
    expected = df[[ 'bid', "price" ]].values
    results = estimator.predict(input_fn = lambda : bidding_data.test_input_fn_for_predict_wp(filenames = filenames))
    predicted = np.array([ result for result in results ])
    index = 0
    while index < predicted.size:
        print('Expected')
        print(expected[index])
        print('Predicted')
        print(predicted[index])
        index = index + 1

def test_on_datasets():
    estimator = get_model()
    test_with_print(estimator, CONFIG['PREDICT_WP_DATASET_VAL'], set_name='validation')

def test():
    test_on_datasets()

def main(argv):
    global BATCH_SIZE, NUM_EPOCHS
    args = parser.parse_args(argv[1:])
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.train_steps
    if args.is_test > 0:
        test()
    else:
        tf.logging.set_verbosity(tf.logging.INFO)
        if args.clean > 0:
            shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        train_and_evaluate()

if __name__ == '__main__':
    tf.app.run(main)
