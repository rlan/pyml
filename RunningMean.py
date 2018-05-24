from __future__ import division

import math

class RunningMean:
  """Compute running mean.

  This class computes running mean.
  Recoded from https://www.johndcook.com/blog/skewness_kurtosis/

  Example
  -------

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

  def __call__(self, input):
    """Update running mean with input

    Parameters
    ----------
    input : list or scalar

    Returns
    -------
    self
      Modified self object
    """
    try:
      for scalar in input:
        self.update(scalar)
    except TypeError:
      self.update(input)
    return self

  def __str__(self):
    return "{}".format(self.mean())

  def clear(self):
    """Clear state and running mean.

    Returns
    -------
    self
      Modified self object
    """
    self.n = 0
    self.M1 = 0.0
    return self

  def update(self, input):
    """Update running mean with input

    Parameters
    ----------
    input : scalar

    Returns
    -------
    self
      Modified self object
    """
    self.n += 1
    delta = float(input) - self.M1
    delta_n = delta / self.n
    self.M1 += delta_n
    return self

  def count(self):
    """Return count.

    Returns
    -------
    int
      Number of data points received.
    """
    return self.n

  def mean(self):
    """Return running mean.

    Returns
    -------
    float
      Running mean.
    """
    return self.M1



if __name__ == "__main__":
  import doctest 
  import sys 
  (failure_count, test_count) = doctest.testmod() 
  sys.exit(failure_count) 
