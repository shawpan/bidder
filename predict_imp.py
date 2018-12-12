""" Train, Evaluate and Predict Winning Rate of a Bidding Request """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import shutil
import functools
import adanet
from adanet.examples import simple_dnn

import bidding_data
import config
import pandas as pd
import numpy as np

CONFIG = config.get_config()
OUTPUT_DIR = CONFIG['OUTPUT_DIR_PREDICT_IMP']

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

""" Get the model definition """
def get_model():
    return get_adanet_model()

def ensemble_architecture(result):
  """Extracts the ensemble architecture from evaluation results."""

  architecture = result["architecture/adanet/ensembles"]
  # The architecture is a serialized Summary proto for TensorBoard.
  summary_proto = tf.summary.Summary.FromString(architecture)
  return summary_proto.value[0].tensor.string_val[0]

def get_adanet_model():
    # Estimator configuration.
    runConfig = tf.estimator.RunConfig(
        save_checkpoints_steps=100,
        save_summary_steps=100,
        tf_random_seed=RANDOM_SEED)
    estimator = adanet.Estimator(
        model_dir = OUTPUT_DIR,
        # adanet_loss_decay=0.99,
        head=tf.contrib.estimator.binary_classification_head(),
        subnetwork_generator=simple_dnn.Generator(
            learn_mixture_weights=True,
            dropout=CONFIG["DROPOUT"],
            feature_columns=bidding_data.get_feature_columns_for_imp_prediction(),
            optimizer=tf.train.RMSPropOptimizer(learning_rate=ADANET_LEARNING_RATE),
            seed=RANDOM_SEED),
        max_iteration_steps=NUM_EPOCHS // ADANET_ITERATIONS,
        evaluator=adanet.Evaluator(
            input_fn=lambda : bidding_data.validation_input_fn_for_predict_imp(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS),
            steps=EVAL_STEPS),
        config=runConfig)

    return estimator

""" Train the model """
def train_and_evaluate():
    estimator = get_model()

    train_spec = tf.estimator.TrainSpec(
                       input_fn = lambda : bidding_data.train_input_fn_for_predict_imp(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS),
                       max_steps = NUM_EPOCHS)
    eval_spec = tf.estimator.EvalSpec(
                       input_fn = lambda : bidding_data.validation_input_fn_for_predict_imp(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS),
                       steps = EVAL_STEPS,
                       # exporters=exporter,
                       start_delay_secs = 1, # start evaluating after N seconds
                       throttle_secs = 10)
    results, _  = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    if results is not None:
        print(results)
        print("Accuracy:", results["accuracy"])
        print("Loss:", results["average_loss"])
        print("Architecture:", ensemble_architecture(results))

def test():
    pass

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
