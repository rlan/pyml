"""
Bounding Box object

Examples:
>>> from BoundingBox import BoundingBox
>>> box = BoundingBox( (1,2), (4,5) )
>>> print(box.ul())
[1 2]
>>> print(box.lr())
[4 5]
>>> print(box.area())
9
>>> print(box.isSmall(threshold=16))
True
"""

from __future__ import print_function
from __future__ import division

import numpy as np

class BoundingBox:

  def __init__(self, upper_left=None, lower_right=None):
    if upper_left == None:
      self.upper_left_ = np.zeros(1,2)
    else:
      self.upper_left_ = np.array(upper_left)

    if lower_right == None:
      self.lower_right_ = np.zeros(1,2)
    else:      
      self.lower_right_ = np.array(lower_right)

  def area(self):
    width = self.lower_right_[0] - self.upper_left_[0]
    height = self.lower_right_[1] - self.upper_left_[1]
    return width * height

  def isSmall(self, threshold = 16):
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


if __name__ == "__main__":
    import doctest
    doctest.testmod()
