from __future__ import print_function
from __future__ import division

import numpy as np

class BoundingBox:
  """"Bounding Box object
  - works for pixel values as well as real values.

    >>> import numpy as np
    >>> from BoundingBox import BoundingBox
    >>> box = BoundingBox( (1,2), (4,5) )
    >>> box.ul()
    array([1, 2])
    >>> box.lr()
    array([4, 5])
    >>> box.ur()
    array([4, 2])
    >>> box.ll()
    array([1, 5])
    >>> box.contour()
    array([[1, 2],
           [4, 2],
           [4, 5],
           [1, 5],
           [1, 2]])
    >>> box.area()
    9
    >>> box.isSmall(threshold=16)
    True
    >>> box.bound(lower_right=[3,4]).lr()
    array([3, 4])
    >>> box.bound(upper_left=[2,3]).ul()
    array([2, 3])
    >>> box.area()
    1
    >>> print(box)
    [[2 3]
     [3 4]]

    >>> contour = np.array([[1, 2], [4, 2], [4, 5], [1, 5], [1, 2]])
    >>> print(BoundingBox.fromContour(contour))
    [[ 1.  2.]
     [ 4.  5.]]
  """

  def __init__(self, upper_left=None, lower_right=None):
    """Init object if upper left or lower right point is given.
    Parameters
    ----------
    upper_left : np.array(2)
    lower_right : np.array(2)
    """
    if upper_left is None:
      self.upper_left_ = np.zeros(2)
    else:
      self.upper_left_ = np.array(upper_left)
      #print(type(self.upper_left_), self.upper_left_)

    if lower_right is None:
      self.lower_right_ = np.zeros(2)
    else:      
      self.lower_right_ = np.array(lower_right)
      #print(type(self.lower_right_), self.lower_right_)

  @classmethod
  def fromContour(cls, contour):
    """Create a bounding box from a contour set.

    Parameters
    ----------
    contour : list of np.array(2)
      A list of points describing a contour.

    Returns
    -------
    self
      A new self object.
    """
    mins = np.amin(contour, axis=0)
    maxs = np.amax(contour, axis=0)
    upper_left = np.floor(mins)  # assumes pixel as coordinates
    lower_right = np.floor(maxs) # assumes pixel as coordinates
    return cls(upper_left, lower_right)

  def area(self):
    """
    Returns
    -------
    double
      Area of bounding box.
    """
    width = self.lower_right_[0] - self.upper_left_[0]
    height = self.lower_right_[1] - self.upper_left_[1]
    return width * height

  def isSmall(self, threshold = 16):
    """
    Parameters
    ----------
    threshold : int
      Threshold to compare against.

    Returns
    -------
    bool
      True if area is less than threshold, otherwise False.
    """
    return self.area() < threshold

  def ul(self):
    """
    Returns
    -------
    np.array(2)
      Return the upper left point, e.g. (row, col)
    """
    return self.upper_left_

  def lr(self):
    """
    Returns
    -------
    np.array(2)
      Return the lower right point, e.g. (row, col)
    """
    return self.lower_right_

  def ur(self):
    """
    Returns
    -------
    np.array(2)
      Return the upper right point, e.g. (row, col)
    """
    return np.array([self.lower_right_[0], self.upper_left_[1]])

  def ll(self):
    """
    Returns
    -------
    np.array(2)
      Return the lower left point, e.g. (row, col)
    """
    return np.array([self.upper_left_[0], self.lower_right_[1]])

  def contour(self):
    """
    Returns
    -------
    A np.array of np.array(2)
      Each contour is an ndarray of shape (n, 2), consisting of n (row, column) coordinates along the contour.
    """
    return np.array([self.ul(), self.ur(), self.lr(), self.ll(), self.ul()])

  def bound(self, upper_left = None, lower_right = None):
    """Trim bounding box by given limits.

    Parameters
    ----------
    upper_left : np.array(2)
    lower_right : np.array(2)

    Returns
    -------
    self
      Modified self object.
    """
    if upper_left is not None:
      if upper_left[0] > self.upper_left_[0]:
        self.upper_left_[0] = upper_left[0]
      if upper_left[1] > self.upper_left_[1]:
        self.upper_left_[1] = upper_left[1]

    if lower_right is not None:
      if self.lower_right_[0] > lower_right[0]:
        self.lower_right_[0] = lower_right[0]
      if self.lower_right_[1] > lower_right[1]:
        self.lower_right_[1] = lower_right[1]

    return self

  def __str__(self):
    """Return string representation
    Returns
    -------
    str
      String representation of self.
    """
    x = np.array([self.upper_left_, self.lower_right_])
    return "{}".format(x)


if __name__ == "__main__":
  import doctest 
  import sys 
  (failure_count, test_count) = doctest.testmod() 
  sys.exit(failure_count) 
