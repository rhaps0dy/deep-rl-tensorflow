import async_agent
from ..history import ForwardViewHistory
import numpy as np
import tensorflow as tf

class ForwardViewA3CAgent(async_agent.AsyncAgent):
  def __init__(self, sess, global_network, target_network, env, stat, conf,
               local_network, tid, val_optimizer, act_optimizer, global_t,
               global_t_semaphore, learning_rate_op, entropy_regularization_op,
               value_network=None):
    super(ForwardViewA3CAgent, self).__init__(
      sess, global_network, local_network, env, stat, conf, tid, global_t,
      global_t_semaphore, learning_rate_op)
    del self.pred_network
    del self.target_network
    self.global_action_network = global_network
    self.action_network = local_network
    if value_network is None:
      self.value_network = self.action_network
      self.global_value_network = self.global_action_network
      self.separate_networks = False
    else:
      self.value_network = value_network
      self.global_value_network = target_network
      self.value_network.create_copy_op(self.global_value_network)
      self.separate_networks = True
    self.entropy_regularization_op = entropy_regularization_op

    self.history = ForwardViewHistory(conf.data_format, conf.history_length,
                                      self.trace_steps, conf.observation_dims)

    with tf.variable_scope('thread_%d' % (self.tid)):
      self.returns_ph = tf.placeholder(tf.float32, [None], name='returns')
      self.actions_ph = tf.placeholder(tf.int64, [None], name='actions')
      actions_one_hot = tf.one_hot(self.actions_ph, self.env.action_size, 1.0,
                                   0.0, axis=-1, dtype=tf.float32,
                                   name='actions_one_hot')
      advantage = self.returns_ph - self.value_network.values
      if conf.entropy_regularization_minimum != 0.0 or conf.entropy_regularization != 0.0:
        action_logs = tf.log(self.action_network.actions)
        taken_action_p_log = tf.reduce_sum(
          tf.mul(action_logs, actions_one_hot), reduction_indices=1)
        entropy = -tf.reduce_sum(
          tf.mul(self.action_network.actions, action_logs), reduction_indices=1)
        loss_actions = tf.mul(taken_action_p_log, advantage) + self.entropy_regularization_op * entropy
      else:
        taken_action_p = tf.reduce_sum(
          tf.mul(self.action_network.actions, actions_one_hot), reduction_indices=1)
        loss_actions = tf.mul(tf.log(taken_action_p), advantage)
      self.loss_actions = -tf.reduce_mean(loss_actions)
      loss_values = tf.square(advantage)
      self.loss_values = tf.reduce_mean(loss_values)

      grads_values = self.create_grads(self.loss_values,
        {'act_w', 'act_b'}, self.value_network, self.global_value_network)
      grads_actions = self.create_grads(self.loss_actions,
        {'val_w', 'val_b'}, self.action_network, self.global_action_network)

      self.val_optim = val_optimizer.apply_gradients(grads_values)
      self.act_optim = act_optimizer.apply_gradients(grads_actions)

      self.returns = np.zeros([self.trace_steps], dtype=np.float32)
      self.actions = np.zeros([self.trace_steps], dtype=np.int64)
      self.possible_actions = np.array(
        range(self.action_network.actions.get_shape().as_list()[1]), dtype=np.int64)

  def train(self):
    terminal = True
    while self.cont[0] and self.global_t[0] <= self.max_t:
      self.action_network.run_copy()
      if self.separate_networks:
        self.value_network.run_copy()
      if terminal:
        observation, _, _ = self.new_game()
        self.history.fill(observation)
      self.history.advance()
      t = 0
      reward = 0.
      terminal = False
      while t < self.trace_steps and not terminal:
        policy = self.action_network.calc('actions', [self.history.get(t)])
        self.actions[t] = np.random.choice(self.possible_actions, p=policy.flatten())
        observation, r, terminal, _ = \
          self.env.step(self.actions[t], is_training=True)
        reward += r
        self.returns[t] = r
        self.history.add(observation)
        t += 1
      if terminal:
        value = 0.
      else:
        value = self.value_network.calc('values', [self.history.get(t)])

      real_time_steps = t
      t -= 1
      while t >= 0:
        self.returns[t] += self.discount_r * value
        value = self.returns[t]
        t -= 1

      dc = {self.action_network.inputs: self.history.get_all(),
            self.value_network.inputs: self.history.get_all(),
          self.returns_ph: self.returns[:real_time_steps],
          self.actions_ph: self.actions[:real_time_steps]}
      if self.global_t[0] < self.t_learn_start:
        _, loss_value, loss_action, q, entropy_regularization = self.sess.run(
          [self.val_optim, self.loss_values, self.loss_actions,
           self.value_network.values, self.entropy_regularization_op], dc)
      else:
        _, _, loss_value, loss_action, q, entropy_regularization = self.sess.run(
          [self.val_optim, self.act_optim, self.loss_values, self.loss_actions,
           self.value_network.values, self.entropy_regularization_op], dc)

      self.advance_t(real_time_steps, self.actions[:real_time_steps], reward,
                     terminal, entropy_regularization, q, loss_value, True,
                     self.learning_rate_op, real_time_steps, loss_action)
