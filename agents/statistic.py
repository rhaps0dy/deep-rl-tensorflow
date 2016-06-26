import os
import numpy as np
import tensorflow as tf

class Statistic(object):
  def __init__(self, sess, t_test, t_learn_start, trace_steps, model_dir,
               variables, max_to_keep=20):
    self.sess = sess
    self.t_test = t_test
    self.iteration_goal = t_test-1
    self.t_learn_start = t_learn_start

    self.reset()
    self.max_avg_ep_reward = 0

    with tf.variable_scope('t'):
      self.t_op = tf.Variable(0, trainable=False, name='t')
      self.t_add_op = [None]
      for i in xrange(1, trace_steps+1):
        self.t_add_op.append(self.t_op.assign_add(i))

    self.model_dir = model_dir
    self.saver = tf.train.Saver(variables + [self.t_op], max_to_keep=max_to_keep)
    self.writer = tf.train.SummaryWriter('./logs/%s' % self.model_dir, self.sess.graph)

    with tf.variable_scope('summary'):
      scalar_summary_tags = [
        'average/reward', 'average/value loss', 'average/action loss', 'average/q',
        'episode/max reward', 'episode/min reward', 'episode/avg reward',
        'episode/num of game', 'training/learning_rate', 'training/epsilon',
      ]

      self.summary_placeholders = {}
      self.summary_ops = {}

      for tag in scalar_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.scalar_summary(tag, self.summary_placeholders[tag])

      histogram_summary_tags = ['episode/rewards', 'episode/actions']

      for tag in histogram_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.histogram_summary(tag, self.summary_placeholders[tag])


  def reset(self):
    self.num_game = 0
    self.update_count = 0
    self.ep_reward = 0.
    self.total_loss_value = 0.
    self.total_loss_action = 0.
    self.total_reward = 0.
    self.actions = []
    self.total_q = []
    self.ep_rewards = []

  def on_step(self, t, action, reward, terminal, ep, q, loss_value, is_update,
              learning_rate_op, time_steps=1, loss_action=0.0):
    if t >= self.t_learn_start:
      self.total_q.extend(q)
      self.actions.extend(action)

      self.total_loss_value += loss_value
      self.total_loss_action += loss_action
      self.total_reward += reward

      self.ep_reward += reward
      if terminal:
        self.num_game += 1
        self.ep_rewards.append(self.ep_reward)
        self.ep_reward = 0.

      if is_update:
        self.update_count += 1

      if t >= self.iteration_goal and self.update_count != 0:
        self.iteration_goal += self.t_test

        avg_q = np.mean(self.total_q)
        avg_loss_value = self.total_loss_value / self.update_count
        avg_loss_action = self.total_loss_action / self.update_count
        avg_reward = self.total_reward / self.t_test

        try:
          max_ep_reward = np.max(self.ep_rewards)
          min_ep_reward = np.min(self.ep_rewards)
          avg_ep_reward = np.mean(self.ep_rewards)
        except:
          max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

        print('\navg_r: %.4f, avg_l_v: %.6f, avg_l_a: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' %
            (avg_reward, avg_loss_value, avg_loss_action, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, self.num_game))

        if self.max_avg_ep_reward * 0.9 <= avg_ep_reward:
          assert t == self.get_t()
          self.save_model(t)
          self.max_avg_ep_reward = max(self.max_avg_ep_reward, avg_ep_reward)

        self.inject_summary({
            'average/q': avg_q,
            'average/value loss': avg_loss_value,
            'average/action loss': avg_loss_action,
            'average/reward': avg_reward,
            'episode/max reward': max_ep_reward,
            'episode/min reward': min_ep_reward,
            'episode/avg reward': avg_ep_reward,
            'episode/num of game': self.num_game,
            'episode/actions': self.actions,
            'episode/rewards': self.ep_rewards,
            'training/learning_rate': learning_rate_op.eval(session=self.sess),
            'training/epsilon': ep,
          }, t)

        self.reset()

    self.t_add_op[time_steps].eval(session=self.sess)

  def inject_summary(self, tag_dict, t):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, t)

  def get_t(self):
    return self.t_op.eval(session=self.sess)

  def save_model(self, t):
    print(" [*] Saving checkpoints...")
    model_name = type(self).__name__

    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
    self.saver.save(self.sess, self.model_dir, global_step=t)

  def load_model(self):
    ckpt = tf.train.get_checkpoint_state(self.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      fname = os.path.join(self.model_dir, ckpt_name)
      self.saver.restore(self.sess, fname)
      t = self.get_t()
      while self.iteration_goal < t:
        self.iteration_goal += self.t_test
      print(" [*] Load SUCCESS: %s" % fname)
      return True
    else:
      print(" [!] Load FAILED: %s" % self.model_dir)
      return False
