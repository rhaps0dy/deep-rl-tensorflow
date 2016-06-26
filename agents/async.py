import os
import time
import numpy as np
import tensorflow as tf
import threading
from logging import getLogger
from tqdm import tqdm

from .agent import Agent
from .history import ForwardViewHistory

logger = getLogger(__name__)

class Async:
  def __init__(self, sess, global_network, env, stat, conf, target_network):
    learning_rate_op = tf.maximum(conf.learning_rate_minimum,
        tf.train.exponential_decay(
            conf.learning_rate,
            stat.t_op,
            conf.learning_rate_decay_step,
            conf.learning_rate_decay,
            staircase=True))
    optimizer = tf.train.RMSPropOptimizer(learning_rate_op,
                                          momentum=0.95, epsilon=0.01, use_locking=True)
    self.stat = stat
    self.global_t = [0]
    self.global_t_semaphore = threading.Semaphore(1)

    self.agents = []
    thread_id = 0
    for nn in target_network:
      self.agents.append(A3CAgent(sess, global_network, env, stat, conf,
                                  local_network=nn, tid=thread_id,
                                  global_optimizer=optimizer,
                                  global_t=self.global_t,
                                  global_t_semaphore=self.global_t_semaphore,
                                  learning_rate_op=learning_rate_op))
      thread_id += 1

  def play(self, ep_end):
    self.agents[0].play(ep_end)

  def train(self, t_train_max):
    # Lists are used so child threads can modify the elements
    tf.initialize_all_variables().run()
    self.stat.load_model()
    self.global_t[0] = self.stat.get_t()
    try:
      threads = []
      for a in self.agents[1:]:
        ta = threading.Thread(target=lambda: a.train(t_train_max))
        threads.append(ta)
        ta.start()
      self.agents[0].train(t_train_max)
    except KeyboardInterrupt:
      self.global_t_semaphore.acquire()
      self.global_t[0] = t_train_max
      self.global_t_semaphore.release()
    for t in threads:
      t.join()

class A3CAgent(Agent):
  def __init__(self, sess, global_network, env, stat, conf, local_network, tid,
               global_optimizer, global_t, global_t_semaphore, learning_rate_op):
    super(A3CAgent, self).__init__(sess, global_network, env, stat, conf, target_network=local_network)
    self.tid = tid
    self.global_optimizer = global_optimizer
    self.global_t = global_t
    self.global_t_semaphore = global_t_semaphore
    self.global_network = global_network
    self.network = local_network
    self.trace_steps = conf.trace_steps
    self.learning_rate_op = learning_rate_op

    self.history = ForwardViewHistory(conf.data_format, conf.history_length,
                                      self.trace_steps, conf.observation_dims)

    with tf.variable_scope('thread_%d' % (self.tid)):
      self.returns_ph = tf.placeholder(tf.float32, [None], name='returns')
      self.actions_ph = tf.placeholder(tf.int64, [None], name='actions')
      actions_one_hot = tf.one_hot(self.actions_ph, self.env.action_size, 1.0, 0.0, axis=-1, dtype=tf.float32, name='actions_one_hot')
      taken_action_p = tf.reduce_sum(
        tf.mul(self.network.actions, actions_one_hot), reduction_indices=1)
      advantage = self.returns_ph - self.network.value
      loss_actions = tf.mul(tf.log(taken_action_p), advantage)
      self.loss_actions = -tf.reduce_sum(loss_actions)
      loss_values = tf.square(advantage)
      self.loss_values = tf.reduce_sum(loss_values)

      def grads(loss, exclude):
        vs = list(set(self.network.var.keys()) - exclude)
        gs = tf.gradients(loss, [self.network.var[v] for v in vs])
        return dict(zip(vs, gs))

      grads_actions = grads(self.loss_actions, {'val_b', 'val_w'})
      grads_values = grads(self.loss_values, {'act_b', 'act_w'})

      grads_and_vars = []
      for k in set(grads_values.keys() + grads_actions.keys()):
        if k in grads_actions:
          if k in grads_values:
            grad = tf.add(grads_actions[k], grads_values[k])
          else:
            grad = grads_actions[k]
        else:
          grad = grads_values[k]
        if self.max_grad_norm > 0.:
          grad = tf.clip_by_norm(grad, self.max_grad_norm)
        grads_and_vars.append((grad, self.global_network.var[k]))
      self.optim = self.global_optimizer.apply_gradients(grads_and_vars)

      self.returns = np.zeros([self.trace_steps], dtype=np.float32)
      self.actions = np.zeros([self.trace_steps], dtype=np.int64)
      self.possible_actions = \
        np.array(range(self.network.actions.get_shape().as_list()[1]), dtype=np.int64)


  def train(self, max_t):
    # 0. Prepare training
    observation, _, terminal = self.new_game()
    self.history.reset()
    self.history.fill(observation)

    if self.tid == 0:
      progress_bar = tqdm(total=max_t)
      prev_update_t = update_t = 0

    while self.global_t[0] <= max_t:
      self.network.run_copy()
      self.history.advance()
      t = 0
      reward = 0.
      terminal = False
      while t < self.trace_steps and not terminal:
        policy = self.network.calc('actions', [self.history.get(t)])
        self.actions[t] = np.random.choice(self.possible_actions, p=policy.flatten())
        observation, r, terminal, _ = \
          self.env.step(self.actions[t], is_training=True)
        reward += r
        self.returns[t] = r
        self.history.add(observation)
        t += 1
      if terminal:
        value = 0.
        observation, _, _ = self.new_game()
      else:
        value = self.network.calc('value', [self.history.get(t)])

      real_time_steps = t
      t -= 1
      while t >= 0:
        self.returns[t] += self.discount_r * value
        value = self.returns[t]
        t -= 1

      _, loss_value, loss_action, q = self.sess.run(
        [self.optim, self.loss_values, self.loss_actions, self.network.value],
        {self.network.inputs: self.history.get_all(),
         self.returns_ph: self.returns[:real_time_steps],
         self.actions_ph: self.actions[:real_time_steps]})

      self.global_t_semaphore.acquire()
      if self.stat:
        self.stat.on_step(self.global_t[0], self.actions[:real_time_steps],
                          reward, terminal, 0, q, loss_value, True,
                          self.learning_rate_op, real_time_steps, loss_action)
      self.global_t[0] += real_time_steps
      gt = self.global_t[0]
      self.global_t_semaphore.release()

      if self.tid == 0:
        prev_update_t = update_t
        update_t = gt
        progress_bar.update(update_t - prev_update_t)
