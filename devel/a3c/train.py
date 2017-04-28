#! /usr/bin/env python
import unittest
import gym
import sys
import os
import numpy as np
import tensorflow as tf
import itertools
import shutil
import threading
import multiprocessing

from inspect import getsourcefile
# current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
# import_path = os.path.abspath(os.path.join(current_path, "../.."))

from os import path
#sys.path.append("../")
#from env import maze

# if import_path not in sys.path:
  # sys.path.append(import_path)

# from lib.atari import helpers as atari_helpers
from estimators import ValueEstimator, PolicyEstimator
from policy_monitor import PolicyMonitor
from worker import Worker


tf.flags.DEFINE_string("model_dir", "/tmp/a3c", 
                       "Directory to write Tensorboard summaries and videos to.")
tf.flags.DEFINE_string("env", "PongNoFrameskip-v3", 
                       "Name of gym Atari environment, e.g. Breakout-v0")
tf.flags.DEFINE_integer("t_max", 5, 
                        "Number of steps before performing an update")
tf.flags.DEFINE_integer("n_frame", 4, 
                        "Number of history frames as observations")
tf.flags.DEFINE_integer("max_global_steps", None, 
                        "Stop training after this many steps in " + \
                        "the environment. Defaults to running indefinitely.")
tf.flags.DEFINE_integer("eval_every", 300, 
                        "Evaluate the policy every N seconds")
tf.flags.DEFINE_float("checkpoint_hour", 0.5, 
                        "Save trained NN every N hours")
tf.flags.DEFINE_boolean("reset", False, 
                        "If set, delete the existing model directory and " + \
                        " start training from scratch.")
tf.flags.DEFINE_integer("parallelism", None, 
                        "Number of threads to run. If not set " + \
                        "we run [num_cpu_cores] threads.")

FLAGS = tf.flags.FLAGS


def main():

  # Depending on the game we may have a limited action space
  env_ = gym.make(FLAGS.env)
  num_actions = env_.action_space.n
  dim_obs = list(env_.observation_space.shape)
  assert len(dim_obs) == 3 and dim_obs[2] == 3 #make sure it is a RGB frame
  N_FRAME = FLAGS.n_frame if FLAGS.n_frame else 1
  dim_obs[2] *= N_FRAME
  print("Valid number of actions is {}".format(num_actions))
  print("The dimension of the observation space is {}".format(dim_obs))
  env_.close()

  # Set the number of workers
  NUM_WORKERS = (FLAGS.parallelism 
                 if FLAGS.parallelism else 
                 multiprocessing.cpu_count())

  MODEL_DIR = FLAGS.model_dir
  CP_H = FLAGS.checkpoint_hour
  CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
  TENSORBOARD_DIR = os.path.join(MODEL_DIR, "tb")

  # Optionally empty model directory
  if FLAGS.reset:
    shutil.rmtree(MODEL_DIR, ignore_errors=True)

  if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)


  summary_writer = tf.summary.FileWriter(TENSORBOARD_DIR)

  with tf.device("/cpu:0"):

    # Keeps track of the number of updates we've performed
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Global policy and value nets
    with tf.variable_scope("global") as vs:
      policy_net = PolicyEstimator(
        num_outputs=num_actions, 
        dim_inputs = dim_obs
      )
      value_net = ValueEstimator(
        reuse=True, 
        dim_inputs = dim_obs
      )

    # Global step iterator
    global_counter = itertools.count()

    # Create worker graphs
    workers = []
    for worker_id in range(NUM_WORKERS):
      # We only write summaries in one of the workers because they're
      # pretty much identical and writing them on all workers
      # would be a waste of space
      worker_summary_writer = None
      if worker_id == 0:
        worker_summary_writer = summary_writer

      worker = Worker(
        name="worker_{}".format(worker_id),
        env=gym.make(FLAGS.env),
        policy_net=policy_net,
        value_net=value_net,
        global_counter=global_counter,
        discount_factor = 0.99,
        summary_writer=worker_summary_writer,
        max_global_steps=FLAGS.max_global_steps,
        n_frame = N_FRAME
      )
      workers.append(worker)

    saver = tf.train.Saver(
      keep_checkpoint_every_n_hours=CP_H, 
      max_to_keep=10
    )

    # Used to occasionally save videos for our policy net
    # and write episode rewards to Tensorboard
    pe = PolicyMonitor(
      env=gym.make(FLAGS.env),
      policy_net=policy_net,
      summary_writer=summary_writer,
      saver=saver)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()

    # Load a previous checkpoint if it exists
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
      print("Loading model checkpoint: {}".format(latest_checkpoint))
      saver.restore(sess, latest_checkpoint)

    # Start worker threads
    worker_threads = []
    for worker in workers:
      print("starting worker:")
      worker_fn = lambda: worker.run(sess, coord, FLAGS.t_max)
      t = threading.Thread(target=worker_fn)
      t.start()
      worker_threads.append(t)

    # Start a thread for policy eval task
    monitor_thread = threading.Thread(
      target=lambda: pe.continuous_eval(FLAGS.eval_every, sess, coord)
    )
    monitor_thread.start()

    # Wait for all workers to finish
    coord.join(worker_threads)


if __name__ == '__main__':
  main()
