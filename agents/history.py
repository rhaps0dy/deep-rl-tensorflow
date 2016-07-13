import numpy as np

class History(object):
  def __init__(self, data_format, batch_size, history_length, screen_dims):
    self.data_format = data_format
    self.history = np.zeros([history_length] + screen_dims, dtype=np.float32)
    if self.data_format == 'NHWC' and len(self.history.shape) == 3:
      self.history_transposed = np.transpose(self.history, (1, 2, 0))
    else:
      self.history_transposed = self.history
    self.reset()

  def add(self, screen):
    self.history[:-1] = self.history[1:]
    self.history[-1] = screen

  def reset(self):
    self.history *= 0

  def get(self):
    return self.history_transposed

class ForwardViewHistory(History):
  def __init__(self, data_format, history_length, trace_steps, screen_dims):
    super(ForwardViewHistory, self).__init__(data_format, None,
                                             history_length+trace_steps,
                                             screen_dims)
    self.history_length = history_length
    self.trace_steps = trace_steps
    if self.data_format == 'NHWC' and len(self.history.shape) == 3:
      f = lambda a: np.transpose(a, (1,2,0))
    else:
      f = lambda a: a
    self.trace_slices = [f(self.history[i:i+history_length,...]) for \
                          i in xrange(self.trace_steps+1)]
    self.reset()

  def reset(self):
    self.history *= 0
    self.counter = 0

  def fill(self, screen):
    self.history[:] = screen

  def add(self, screen):
    self.history[self.counter] = screen
    self.counter += 1

  def advance(self):
    self.trace_slices[0][...] = self.trace_slices[self.counter-self.history_length]
    self.counter = self.history_length

  def get(self, t=0):
    return self.trace_slices[t]

  def get_all(self):
    return self.trace_slices[:self.counter-self.history_length]
