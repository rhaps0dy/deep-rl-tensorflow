import async_agent
from ..history import ForwardViewHistory
import numpy as np
import tensorflow as tf

class ForwardViewDQNAgent(async_agent.AsyncAgent):
  def __init__(self, sess, global_network, target_network, env, stat, conf,
               local_network, tid, optimizer, global_t, global_t_semaphore,
               learning_rate_op):
    super(ForwardViewDQNAgent, self).__init__(
      sess, global_network, local_network, env, stat, conf, tid, global_t,
      global_t_semaphore, learning_rate_op)

    del self.pred_network
    self.target_network = target_network
    self.global_network = global_network
    self.network = local_network

    self.history = ForwardViewHistory(conf.data_format, conf.history_length,
                                      self.trace_steps, conf.observation_dims)

    with tf.variable_scope('thread_%d' % (self.tid)):
      self.returns_ph = tf.placeholder(tf.float32, [None], name='returns')
      self.actions_ph = tf.placeholder(tf.int64, [None], name='actions')
      actions_one_hot = tf.one_hot(self.actions_ph, self.env.action_size, 1.0,
                                   0.0, axis=-1, dtype=tf.float32,
                                   name='actions_one_hot')
      pred_q = tf.reduce_sum(self.network.outputs * actions_one_hot,
                             reduction_indices=1, name='q_acted')
      self.loss = tf.reduce_mean(tf.square(self.returns_ph - pred_q))

      self.grads_and_vars = self.create_grads(
        self.loss, set(), self.network, self.global_network)
      self.optim = optimizer.apply_gradients(self.grads_and_vars)

      if self.double_q:
        self.value = tf.gather(self.target.outputs, self.network.actions)
      else:
        self.value = tf.reduce_max(self.network.outputs, reduction_indices=1)
      self.q_val_mean = tf.reduce_mean(self.network.outputs, reduction_indices=1)

      self.returns = np.zeros([self.trace_steps], dtype=np.float32)
      self.actions = np.zeros([self.trace_steps], dtype=np.int64)

  def train(self):
    terminal = True
    while self.cont[0] and self.global_t[0] <= self.max_t:
      epsilon = self.calc_epsilon(self.global_t[0])
      self.global_t_semaphore.acquire()
      if self.global_t[0] >= self.global_t[1]:
        self.target_network.run_copy()
        self.global_t[1] = (self.global_t[0] // self.t_target_q_update_freq + 1) * \
                            self.t_target_q_update_freq
      self.global_t_semaphore.release()
      self.network.run_copy()
      if terminal:
        observation, _, _ = self.new_game()
        self.history.fill(observation)
      self.history.advance()
      t = 0
      reward = 0.
      terminal = False
      while t < self.trace_steps and not terminal:
        if np.random.random() < epsilon:
          self.actions[t] = np.random.randint(self.env.action_size)
        else:
          self.actions[t] = self.network.calc('actions', [self.history.get(t)])
        observation, r, terminal, _ = \
          self.env.step(self.actions[t], is_training=True)
        reward += r
        self.returns[t] = r
        self.history.add(observation)
        t += 1
      if terminal:
        value = 0.
      else:
        states = [self.history.get(t)]
        value = self.value.eval(
          {self.network.inputs: states, self.target_network.inputs: states},
          session=self.sess)

      real_time_steps = t
      t -= 1
      while t >= 0:
        self.returns[t] += self.discount_r * value
        value = self.returns[t]
        t -= 1

      _, loss, q = self.sess.run([self.optim, self.loss, self.q_val_mean],
                                 {self.network.inputs: self.history.get_all(),
                                  self.returns_ph: self.returns[:real_time_steps],
                                  self.actions_ph: self.actions[:real_time_steps]})

      self.advance_t(real_time_steps, self.actions[:real_time_steps], reward,
                     terminal, epsilon, q, loss, True,
                     self.learning_rate_op, real_time_steps, 0)

