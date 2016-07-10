import tensorflow as tf
import threading
from logging import getLogger

from ..agent import Agent
from ..history import ForwardViewHistory
from .forward_view_dqn import ForwardViewDQNAgent
from .forward_view_a3c import ForwardViewA3CAgent

logger = getLogger(__name__)

class Async:
  def __init__(self, sess, global_network, target_network, envs, stat, conf,
               pred_networks, value_networks=None):
    learning_rate_op = tf.maximum(conf.learning_rate_minimum,
        tf.train.exponential_decay(
            conf.learning_rate,
            stat.t_op,
            conf.learning_rate_decay_step,
            conf.learning_rate_decay,
            staircase=True))
    entropy_regularization_op = tf.maximum(conf.entropy_regularization_minimum,
        tf.train.exponential_decay(
            conf.entropy_regularization,
            stat.t_op,
            conf.entropy_regularization_decay_step,
            conf.entropy_regularization_decay,
            staircase=False))
    val_optimizer = tf.train.RMSPropOptimizer(
      learning_rate_op, decay=conf.decay, momentum=conf.momentum,
      epsilon=conf.rmsprop_epsilon, use_locking=True)
    act_optimizer = tf.train.RMSPropOptimizer(
      learning_rate_op, decay=conf.decay, momentum=conf.momentum,
      epsilon=conf.rmsprop_epsilon, use_locking=True)
    self.stat = stat
    self.global_t = [0, 0]
    self.global_t_semaphore = threading.Semaphore(1)

    self.agents = []
    target_network.create_copy_op(global_network)
    thread_id = 0
    for nn, env in zip(pred_networks, envs):
      if conf.network_output_type in ['normal', 'dueling']:
        self.agents.append(ForwardViewDQNAgent(
          sess, global_network, target_network, env, stat, conf, nn, thread_id,
          val_optimizer, self.global_t, self.global_t_semaphore,
          learning_rate_op))
      elif conf.network_output_type in ['actor_critic']:
        if value_networks is not None:
          self.agents.append(ForwardViewA3CAgent(
            sess, global_network, target_network, env, stat, conf, nn,
            thread_id, val_optimizer, act_optimizer, self.global_t,
            self.global_t_semaphore, learning_rate_op,
            entropy_regularization_op,
            value_network=value_networks[thread_id]))
        else:
          self.agents.append(ForwardViewA3CAgent(
            sess, global_network, global_network, env, stat, conf, nn,
            thread_id, val_optimizer, act_optimizer, self.global_t,
            self.global_t_semaphore, learning_rate_op,
            entropy_regularization_op))
      else:
        raise ValueError("Unknown network_output_type: %s" % conf.network_output_type)
      thread_id += 1

  def play(self, ep_end):
    self.agents[0].play(ep_end)

  def train(self, t_train_max):
    # Lists are used as shared thread memory
    tf.initialize_all_variables().run()
    self.stat.load_model()
    self.global_t[0] = self.stat.get_t()
    cont = [True]
    try:
      threads = []
      for a in self.agents[1:]:
        a.train_prepare(t_train_max, cont)
        ta = threading.Thread(target=a.train)
        threads.append(ta)
        ta.start()
      self.agents[0].train_prepare(t_train_max, cont)
      self.agents[0].train()
    except KeyboardInterrupt:
      cont[0] = False
    for t in threads:
      t.join()
