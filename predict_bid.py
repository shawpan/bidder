""" Train, Evaluate and Predict Winning Rate of a Bidding Request """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
from tensorflow.python.ops import math_ops

import shutil
import functools
import adanet
from adanet.examples import simple_dnn

import bidding_data
import config
import pandas as pd
import numpy as np
# import scipy.stats

CONFIG = config.get_config()
OUTPUT_DIR = CONFIG['OUTPUT_DIR_PREDICT_BID']

BATCH_SIZE = CONFIG['BATCH_SIZE'] # 512
NUM_EPOCHS = CONFIG['NUM_EPOCHS'] # 4000
EVAL_STEPS = CONFIG['EVAL_STEPS'] # 100
ADANET_LEARNING_RATE = CONFIG['ADANET_LEARNING_RATE']
ADANET_ITERATIONS = CONFIG['ADANET_ITERATIONS']
RANDOM_SEED = CONFIG['RANDOM_SEED']

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
    # labels = math_ops.to_float(labels)
    # logits = math_ops.to_float(logits)
    wons = tf.reshape(features['won'], [-1, len(CONFIG['PREDICT_BID_LABELS'])])
    error = labels - logits
    # mean, variance = tf.nn.moments(error, [0])
    # error = ( error - mean ) / variance
    cdf = tf.distributions.Normal(loc=0.0, scale=1.0).cdf(error)
    # scipy.stats.norm.logcdf(tf.math.abs(labels - logits))
    # print(cdf)
    loss_on_losts = (1. - wons) * tf.log(1.5 - cdf)
    loss_on_won = wons * tf.log(cdf)
    loss = loss_on_won + loss_on_losts

    return -loss

def get_adanet_model():
    # Estimator configuration.
    runConfig = tf.estimator.RunConfig(
        save_checkpoints_steps=100,
        save_summary_steps=100,
        tf_random_seed=RANDOM_SEED)
    estimator = adanet.Estimator(
        model_dir = OUTPUT_DIR,
        # adanet_loss_decay=0.99,
        head=tf.contrib.estimator.regression_head(
        label_dimension=len(CONFIG['PREDICT_BID_LABELS']),
        loss_fn=first_price_auction_loss,
        loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE
        ),
        subnetwork_generator=simple_dnn.Generator(
            learn_mixture_weights=True,
            dropout=CONFIG["DROPOUT"],
            feature_columns=bidding_data.get_feature_columns_for_bid_prediction(),
            optimizer=tf.train.RMSPropOptimizer(learning_rate=ADANET_LEARNING_RATE),
            seed=RANDOM_SEED),
        max_iteration_steps=NUM_EPOCHS // ADANET_ITERATIONS,
        evaluator=adanet.Evaluator(
            input_fn=lambda : bidding_data.validation_input_fn_for_predict_bid(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS),
            steps=EVAL_STEPS),
        config=runConfig)

    return estimator

""" Train the model """
def train_and_evaluate():
    estimator = get_model()
    train_spec = tf.estimator.TrainSpec(
                       input_fn = lambda : bidding_data.train_input_fn_for_predict_bid(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS),
                       max_steps = NUM_EPOCHS)
    eval_spec = tf.estimator.EvalSpec(
                       input_fn = lambda : bidding_data.validation_input_fn_for_predict_bid(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS),
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
    df = pd.concat((pd.read_csv(f, sep=CONFIG['CSV_SEPARATOR'], na_values=["null"], dtype=dtypes) for f in filenames))
    expected = df[[ 'req_bid', "price", "won", "price" ]].values
    results = estimator.predict(input_fn = lambda : bidding_data.test_input_fn_for_predict_bid(filenames = filenames))
    predicted = np.array([ result for result in results ])
    index = 0
    validWins = 0;
    validLoss = 0;
    numberOfWins = 0
    numberOfLosts = 0
    error = 0;
    neg = 0;
    while index < predicted.size:
        print('Expected')
        print(expected[index])
        print('Predicted')
        print(predicted[index])
        predictedPrice = predicted[index]['predictions'][0]
        if predictedPrice <= 0:
            neg = neg + 1
        if expected[index][2] > 0:
            numberOfWins = numberOfWins + 1
            error = error + np.absolute(expected[index][1] - predictedPrice)
        else:
            numberOfLosts = numberOfLosts + 1
        if expected[index][2] > 0 and expected[index][0] >= predictedPrice: # won
            validWins = validWins + 1
        if expected[index][2] < 1 and expected[index][0] < predictedPrice: # lost
            validLoss = validLoss + 1
        index = index + 1
    print("valid on wins ", (validWins) * 100.0 / numberOfWins, " %")
    print("valid on lost ", (validLoss) * 100.0 / numberOfLosts, " %")
    print("valid ", (validWins + validLoss) * 100.0 / predicted.size, " %")
    print("negative ", neg)
    print("error ", error)

def test_on_datasets():
    estimator = get_model()
    test_with_print(estimator, CONFIG['PREDICT_BID_DATASET_VAL'], set_name='validation')

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
