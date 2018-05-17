"""
Recoded from https://www.johndcook.com/blog/skewness_kurtosis/

>>> from RunningStats import RunningStats
>>> s = RunningStats()
>>> s.count()
0
>>> s([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
>>> s.mean()
5.0
>>> s.variance()
7.5
>>> import math
>>> s.std() == math.sqrt(7.5)
True
>>> s.skewness()
0.0

>>> s.kurtosis()
True

kurtosis == 1.573333 = 157/100 + 1/300
>>> s.kurtosis() == (1.57 + 1.0/300.0)
True

>>> s.min()
1.0
>>> s.max()
9.0
>>> s.count()
9
"""

from __future__ import division

import math

class RunningStats:
  
  def __init__(self):
    self.clear()

  def clear(self):
    self.n = 0
    self.M1 = self.M2 = self.M3 = self.M4 = 0.0
    self.min_ = self.max_ = 0.0
    self.once_ = False
    return self

  def __call__(self, x):
    for v in x:
      self.update(v)
    return self

  def update(self, x):
    n1 = self.n
    self.n += 1
    delta = float(x) - self.M1
    delta_n = delta / self.n
    delta_n2 = delta_n * delta_n
    term1 = delta * delta_n * float(n1)
    self.M1 += delta_n
    self.M4 += term1 * delta_n2 * (float(self.n*self.n) - 3.0*self.n + 3.0) + 6.0 * delta_n2 * self.M2 - 4.0 * delta_n * self.M3
    self.M3 += term1 * delta_n * float(self.n - 2) - 3.0 * delta_n * self.M2
    self.M2 += term1

    if self.once_:
      if x < self.min_:
        self.min_ = x
      if x > self.max_:
        self.max_ = x
    else:
      self.min_ = self.max_ = x
      self.once_ = True
    return self

  def count(self):
    return self.n

  def mean(self):
    return self.M1

  def variance(self):
    return self.M2 / (self.n-1.0)

  def std(self):
    return math.sqrt(self.variance())

  def skewness(self):
    return math.sqrt(self.n) * self.M3 / math.pow(self.M2, 1.5)

  def kurtosis(self):
    return float(self.n) * self.M4 / (self.M2*self.M2) - 3.0

  def min(self):
    return self.min_

  def max(self):
    return self.max_


if __name__ == "__main__":
    import doctest
    doctest.testmod()
