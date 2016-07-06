import gym
import random
import logging
import tensorflow as tf

from utils import get_model_dir
from networks.cnn import CNN
from networks.mlp import MLPSmall
from agents.statistic import Statistic
from environments.environment import ToyEnvironment, AtariEnvironment

flags = tf.app.flags

# Deep q Network
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not. gpu use NHWC and gpu use NCHW for data_format')
flags.DEFINE_string('agent_type', 'DQN', 'The type of agent [DQN, A3C]')
flags.DEFINE_boolean('double_q', False, 'Whether to use double Q-learning')
flags.DEFINE_string('network_header_type', 'nips', 'The type of network header [mlp, nature, nips]')
flags.DEFINE_string('network_output_type', 'normal', 'The type of network output [normal, dueling]')

# Environment
flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
flags.DEFINE_integer('n_action_repeat', 4, 'The number of actions to repeat')
flags.DEFINE_integer('max_random_start', 30, 'The maximum number of NOOP actions at the beginning of an episode')
flags.DEFINE_integer('history_length', 4, 'The length of history of observation to use as an input to DQN')
flags.DEFINE_integer('trace_steps', 1, 'The number of steps of eligibility traces')
flags.DEFINE_integer('max_r', +1, 'The maximum value of clipped reward')
flags.DEFINE_integer('min_r', -1, 'The minimum value of clipped reward')
flags.DEFINE_string('observation_dims', '[80, 80]', 'The dimension of gym observation')
flags.DEFINE_boolean('random_start', True, 'Whether to start with random state')

# Training
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('max_delta', None, 'The maximum value of delta')
flags.DEFINE_integer('min_delta', None, 'The minimum value of delta')
flags.DEFINE_float('ep_start', 1., 'The value of epsilon at start in e-greedy')
flags.DEFINE_float('ep_end', 0.1, 'The value of epsilnon at the end in e-greedy')
flags.DEFINE_integer('batch_size', 32, 'The size of batch for minibatch training')
flags.DEFINE_float('max_grad_norm', 40, 'The maximum norm of gradient while updating')
flags.DEFINE_float('discount_r', 0.99, 'The discount factor for reward')
flags.DEFINE_integer('async_threads', 1, 'The number of simultaneous A3C agents')

# Timer
flags.DEFINE_integer('t_train_freq', 4, '')

# Below numbers will be multiplied by scale
flags.DEFINE_integer('scale', 10000, 'The scale for big numbers')
flags.DEFINE_integer('memory_size', 100, 'The size of experience memory (*= scale)')
flags.DEFINE_integer('t_target_q_update_freq', 1, 'The frequency of target network to be updated (*= scale)')
flags.DEFINE_integer('t_test', 1, 'The maximum number of t while training (*= scale)')
flags.DEFINE_integer('t_ep_end', 100, 'The time when epsilon reach ep_end (*= scale)')
flags.DEFINE_integer('t_train_max', 5000, 'The maximum number of t while training (*= scale)')
flags.DEFINE_float('t_learn_start', 5, 'The time when to begin training (*= scale)')
flags.DEFINE_float('learning_rate_decay_step', 5, 'The learning rate of training (*= scale)')
flags.DEFINE_float('entropy_regularization_decay_step', 5, 'The learning rate of training (*= scale)')

# Optimizer
flags.DEFINE_float('learning_rate', 0.025, 'The learning rate of training')
flags.DEFINE_float('learning_rate_minimum', 0.0025, 'The learning rate of training')
flags.DEFINE_float('learning_rate_decay', 0.96, 'The learning rate of training')
flags.DEFINE_float('decay', 0.99, 'Decay of RMSProp optimizer')
flags.DEFINE_float('momentum', 0.0, 'Momentum of RMSProp optimizer')
flags.DEFINE_float('gamma', 0.99, 'Discount factor of return')
flags.DEFINE_float('rmsprop_epsilon', 0.01, 'Epsilon of RMSProp optimizer')
flags.DEFINE_float('entropy_regularization', 0.5,
                   'The regularization parameter for policy entropy in A3C')
flags.DEFINE_float('entropy_regularization_minimum', 0.0, 'The minimum entropy regularization')
flags.DEFINE_float('entropy_regularization_decay', 0.96, 'The entropy regularization decay')

# Debug
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_string('log_level', 'INFO', 'Log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')
flags.DEFINE_string('tag', '', 'The name of tag for a model, only for debugging')

conf = flags.FLAGS

logger = logging.getLogger()
logger.propagate = False
logger.setLevel(conf.log_level)

# set random seed
tf.set_random_seed(conf.random_seed)
random.seed(conf.random_seed)

def main(_):
  # preprocess
  conf.observation_dims = eval(conf.observation_dims)

  for flag in ['memory_size', 't_target_q_update_freq', 't_test',
               't_ep_end', 't_train_max', 't_learn_start',
               'learning_rate_decay_step', 'entropy_regularization_decay_step']:
    setattr(conf, flag, getattr(conf, flag) * conf.scale)
#  for flag in ['learning_rate', 'learning_rate_minimum']:
#    setattr(conf, flag, getattr(conf, flag) / conf.async_threads)

  if conf.use_gpu:
    conf.data_format = 'NCHW'
  else:
    conf.data_format = 'NHWC'

  model_dir = get_model_dir(conf,
      ['use_gpu', 'max_random_start', 'n_worker', 'is_train', 'memory_size',
       't_save', 't_train', 'display', 'log_level', 'random_seed', 'tag', 'scale'])

  device = '/gpu:0' if conf.use_gpu else '/cpu:0'
  # start
  with tf.Session() as sess, tf.device(device):
    if any(name in conf.env_name for name in ['Corridor', 'FrozenLake']) :
      env = ToyEnvironment(conf.env_name, conf.n_action_repeat, conf.max_random_start,
                        conf.observation_dims, conf.data_format, conf.display)
    else:
      env = AtariEnvironment(conf.env_name, conf.n_action_repeat, conf.max_random_start,
                        conf.observation_dims, conf.data_format, conf.display)

    if conf.network_header_type in ['nature', 'nips']:
      NetworkHead = CNN
      args = {'sess': sess,
              'data_format': conf.data_format,
              'history_length': conf.history_length,
              'observation_dims': conf.observation_dims,
              'output_size': env.env.action_space.n,
              'network_output_type': conf.network_output_type}
    elif conf.network_header_type == 'mlp':
      NetworkHead = MLPSmall
      args = {'sess': sess,
              'history_length': conf.history_length,
              'observation_dims': conf.observation_dims,
              'output_size': env.env.action_space.n,
              'hidden_activation_fn': tf.sigmoid,
              'network_output_type': conf.network_output_type}
    else:
      raise ValueError('Unkown network_header_type: %s' % (conf.network_header_type))

    stat = Statistic(sess, conf.t_test, conf.t_learn_start, conf.trace_steps,
                     model_dir)

    if conf.agent_type == 'DQN':
      from agents.deep_q import DeepQ
      pred_network = NetworkHead(name='pred_network', trainable=True, **args)
      stat.create_writer(pred_network.var.values())
      target_network = NetworkHead(name='target_network', trainable=False, **args)
      agent = DeepQ(sess, pred_network, env, stat, conf,
                    target_network=target_network)
    elif conf.agent_type == 'A3C':
      from agents.async import Async
      global_network = NetworkHead(name='global_network', trainable=False, **args)
      stat.create_writer(global_network.var.values())
      target_network = NetworkHead(name='target_network', trainable=False, **args)
      pred_networks = list(
        NetworkHead(name=('pred_network_%d'%i), trainable=False, **args)
        for i in range(conf.async_threads))
      agent = Async(sess, global_network, env, stat, conf,
                    target_network=target_network, pred_networks=pred_networks)
    else:
      raise ValueError('Unkown agent_type: %s' % (conf.agent_type))

    if conf.is_train:
      agent.train(conf.t_train_max)
    else:
      agent.play(conf.ep_end)

if __name__ == '__main__':
  tf.app.run()
