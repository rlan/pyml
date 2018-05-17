from __future__ import division

import math

class RunningMean:
  """
  Compute running mean.
  Recoded from https://www.johndcook.com/blog/skewness_kurtosis/

    >>> from RunningMean import RunningMean
    >>> s = RunningMean()
    >>> s.mean()
    0.0
    >>> s.count()
    0
    >>> print(s([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]))
    5.0
    >>> s.mean()
    5.0
    >>> s.count()
    9
    >>> print(s.clear())
    0.0
    >>> s.mean()
    0.0
    >>> s.count()
    0
    >>> print(s(100))
    100.0
    >>> print(s([0.0]))
    50.0

  """
  
  def __init__(self):
    self.clear()

  def __call__(self, x):
    try:
      for v in x:
        self.update(v)
    except TypeError:
      self.update(x)
    return self

  def __str__(self):
    return "{}".format(self.mean())

  def clear(self):
    self.n = 0
    self.M1 = 0.0
    return self

  def update(self, x):
    self.n += 1
    delta = float(x) - self.M1
    delta_n = delta / self.n
    self.M1 += delta_n

    return self

  def count(self):
    return self.n

  def mean(self):
    return self.M1



if __name__ == "__main__":
    import doctest
    doctest.testmod()
