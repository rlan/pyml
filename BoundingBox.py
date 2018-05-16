"""
Bounding Box object
- works for pixel values as well as real values.

Examples:
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
       [1, 5],
       [4, 5]])
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
[2 3], [3 4]
"""

from __future__ import print_function
from __future__ import division

import numpy as np

class BoundingBox:

  def __init__(self, upper_left=None, lower_right=None):
    if upper_left is None:
      self.upper_left_ = np.zeros(1,2)
    else:
      self.upper_left_ = np.array(upper_left)

    if lower_right is None:
      self.lower_right_ = np.zeros(1,2)
    else:      
      self.lower_right_ = np.array(lower_right)

  def area(self):
    """
    @return: Return area of bounding box
    """
    width = self.lower_right_[0] - self.upper_left_[0]
    height = self.lower_right_[1] - self.upper_left_[1]
    return width * height

  def isSmall(self, threshold = 16):
    """
    @return: Return True if area is less than threshold, otherwise False
    """
    return self.area() < threshold

  def ul(self):
    """
    Return the upper left point
    @return: An array, e.g. (row, col)
    """
    return self.upper_left_

  def lr(self):
    """
    Return the lower right point
    @return: An array, e.g. (row, col)
    """
    return self.lower_right_

  def ur(self):
    """
    Return the upper right point
    @return: An array, e.g. (row, col)
    """
    return np.array([self.lower_right_[0], self.upper_left_[1]])

  def ll(self):
    """
    Return the lower left point
    @return: An array, e.g. (row, col)
    """
    return np.array([self.upper_left_[0], self.lower_right_[1]])

  def contour(self):
    """
    Each contour is an ndarray of shape (n, 2), consisting of n (row, column) coordinates along the contour.
    @return: list of (n,2)-ndarrays
    """
    return np.array([self.ul(), self.ur(), self.ll(), self.lr()])

  def bound(self, upper_left = None, lower_right = None):
    """
    Limit bounding box corners in place.
    @return: Self
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
    return "{}, {}".format(self.upper_left_, self.lower_right_)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
