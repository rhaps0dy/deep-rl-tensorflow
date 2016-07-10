from .. import agent
from tqdm import tqdm
import tensorflow as tf

class AsyncAgent(agent.Agent):
  def __init__(self, sess, global_network, local_network, env, stat, conf, tid,
               global_t, global_t_semaphore, learning_rate_op):
    # Construct the copy operation for the local network
    super(AsyncAgent, self).__init__(sess, global_network, env, stat, conf,
                                     target_network=local_network)
    self.tid = tid
    self.global_t = global_t
    self.global_t_semaphore = global_t_semaphore
    self.trace_steps = conf.trace_steps
    self.learning_rate_op = learning_rate_op
    self.n_threads = conf.async_threads

  def create_grads(self, loss, exclude, network, global_network):
    vs = list(set(network.var.keys()) - exclude)
    gs = tf.gradients(loss, [network.var[v] for v in vs])
    for i in xrange(len(gs)):
      if self.max_grad_norm > 0.:
        gs[i] = tf.clip_by_norm(gs[i], self.max_grad_norm)
      gs[i] /= self.n_threads
    return zip(gs, map(global_network.var.get, vs))

  def train_prepare(self, max_t, cont):
    self.max_t, self.cont = max_t, cont
    if self.tid == 0:
      self.progress_bar = tqdm(total=self.max_t)
      self.prev_update_t = self.update_t = 0

  def advance_t(self, real_time_steps, *args, **kwargs):
    self.global_t_semaphore.acquire()
    if self.stat:
      self.stat.on_step(self.global_t[0], *args, **kwargs)
    self.global_t[0] += real_time_steps
    gt = self.global_t[0]
    self.global_t_semaphore.release()

    if self.tid == 0:
      self.prev_update_t = self.update_t
      self.update_t = gt
      self.progress_bar.update(self.update_t - self.prev_update_t)


