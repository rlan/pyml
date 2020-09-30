from __future__ import print_function
from __future__ import division

import numpy as np

class Line:
  """Line object

  Example
  -------
  >>> from BoundingBox import Line
  >>> l1 = Line(-2, 2)
  >>> l1.length()
  4

  Test cases:
  --- first
  === second

      ======
  --
        --
  ------
  ----------
  --------------
      ----------
              --

  No overlap.
  >>> x = Line.fromOverlap(Line(1,10), Line(10,20))
  >>> print(x)
  [0 0]
  >>> x.length()
  0

  >>> x = Line.fromOverlap(Line(11,12), Line(10,20))
  >>> print(x)
  [11 12]
  >>> x.length()
  1

  Overlap of 2, first left.
  >>> x = Line.fromOverlap(Line(8,12), Line(10,20))
  >>> print(x)
  [10 12]
  >>> x.length()
  2

  Edge overlap
  >>> x = Line.fromOverlap(Line(6,20), Line(10,20))
  >>> print(x)
  [10 20]
  >>> x.length()
  10

  Second completely inside first.
  >>> x = Line.fromOverlap(Line(1,40), Line(20,30))
  >>> print(x)
  [20 30]
  >>> x.length()
  10

  Edge overlap
  >>> x = Line.fromOverlap(Line(10,25), Line(10,20))
  >>> print(x)
  [10 20]
  >>> x.length()
  10

  No overlap
  >>> x = Line.fromOverlap(Line(0,10), Line(-10,-5))
  >>> print(x)
  [0 0]
  >>> x.length()
  0
  """
  def __init__(self, x1, x2):
    self.x1 = x1
    self.x2 = x2
    # TODO error check. x2 >= x1

  def __str__(self):
    return "[{} {}]".format(self.x1, self.x2)

  def length(self):
    return self.x2 - self.x1

  @classmethod
  def fromOverlap(cls, first, second):
    #print("first", first)
    #print("second", second)
    if first.x2 <= second.x1:
      #print("return 0")
      return cls(0, 0)
    elif first.x2 <= second.x2:
      if first.x1 >= second.x1:
        #print("return 1")
        return first
      else:
        #print("return 2")
        return cls(second.x1, first.x2)
    else: # first.x2 > second.x2
      if first.x1 >= second.x2:
        return cls(0, 0)
      elif first.x1 >= second.x1:
        #print("return 3")
        return cls(first.x1, second.x2)
      else: # first.x1 < second.x1
        #print("return 4")
        return second


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
    [[1. 2.]
     [4. 5.]]
  """

  def __init__(self, upper_left=None, lower_right=None):
    """Init object if upper left or lower right point is given.
    Parameters
    ----------
    upper_left : np.array(2)
    lower_right : np.array(2)
    """
    #TODO error check that ul point is instead ul, lr point is indeed lr.
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


  @classmethod
  def fromOverlap(cls, first, second):
    """Create a bounding box of the intercept.

    Parameters
    ----------
    first : BoundingBox
    second : BoundingBox

    Return
    ------
    BoundingBox

    Example
    -------
    >>> from BoundingBox import BoundingBox

    Some overlap
    >>> first = BoundingBox( (2,1), (5,4) )
    >>> second = BoundingBox( (4,3), (8,5) )
    >>> overlap_box = BoundingBox.fromOverlap(first, second)
    >>> print(overlap_box)
    [[4 3]
     [5 4]]

    No overlap
    >>> first = BoundingBox( (2,1), (5,4) )
    >>> second = BoundingBox( (3,5), (5,7) )
    >>> overlap_box = BoundingBox.fromOverlap(first, second)
    >>> print(overlap_box)
    [[0 0]
     [0 0]]

    Share an edge
    >>> first = BoundingBox( (4,3), (8,5) )
    >>> second = BoundingBox( (3,5), (5,7) )
    >>> overlap_box = BoundingBox.fromOverlap(first, second)
    >>> print(overlap_box)
    [[0 0]
     [0 0]]

    Second completely inside first.
    >>> first = BoundingBox( (2,1), (9,7) )
    >>> second = BoundingBox( (4,3), (8,5) )
    >>> overlap_box = BoundingBox.fromOverlap(first, second)
    >>> print(overlap_box)
    [[4 3]
     [8 5]]

    First completely inside second.
    >>> first = BoundingBox( (4,3), (8,5) )
    >>> second = BoundingBox( (2,1), (9,7) )
    >>> overlap_box = BoundingBox.fromOverlap(first, second)
    >>> print(overlap_box)
    [[4 3]
     [8 5]]

    First is sideway inside second on the right
    >>> first = BoundingBox( (6,4), (7,6) )
    >>> second = BoundingBox( (4,3), (8,5) )
    >>> overlap_box = BoundingBox.fromOverlap(first, second)
    >>> print(overlap_box)
    [[6 4]
     [7 5]]

    Corner touching
    >>> first = BoundingBox( (2,1), (5,4) )
    >>> second = BoundingBox( (5,4), (6,5) )
    >>> overlap_box = BoundingBox.fromOverlap(first, second)
    >>> print(overlap_box)
    [[0 0]
     [0 0]]
    """

    # row
    line_first = Line( first.ul()[0], first.lr()[0] )
    line_second = Line( second.ul()[0], second.lr()[0] )
    row_overlap = Line.fromOverlap(line_first, line_second)
    #print("row_overlap", row_overlap)
    # col
    line_first = Line( first.ul()[1], first.lr()[1] )
    line_second = Line( second.ul()[1], second.lr()[1] )
    col_overlap = Line.fromOverlap(line_first, line_second)
    #print("col_overlap", col_overlap)

    if (row_overlap.length() == 0) or (col_overlap.length() == 0):
      return cls( (0,0), (0,0) )
    else:
      return cls( (row_overlap.x1, col_overlap.x1), (row_overlap.x2, col_overlap.x2) )

  def height(self):
    """
    Returns
    -------
    float
      Height of bounding box.

    Example
    -------
    >>> from BoundingBox import BoundingBox
    >>> b = BoundingBox( (4,3), (8,5) )
    >>> b.height()
    4
    """
    return self.lower_right_[0] - self.upper_left_[0]

  def width(self):
    """
    Returns
    -------
    float
      Width of bounding box.

    Example
    -------
    >>> from BoundingBox import BoundingBox
    >>> b = BoundingBox( (4,3), (8,5) )
    >>> b.width()
    2
    """
    return self.lower_right_[1] - self.upper_left_[1]    

  def area(self):
    """
    Returns
    -------
    float
      Area of bounding box.

    Example
    -------
    >>> from BoundingBox import BoundingBox
    >>> b = BoundingBox( (4,3), (8,5) )
    >>> b.area()
    8
    """
    return self.height() * self.width()

  def aspectRatio(self):
    """
    Returns
    -------
    float
      Aspect ratio of the box

    Example
    -------
    >>> from BoundingBox import BoundingBox
    >>> b = BoundingBox( (4,3), (8,5) )
    >>> b.aspectRatio()
    0.5

    >>> b = BoundingBox( (4,3), (4,5) )
    >>> b.aspectRatio()
    nan
    """
    if float(self.height()) == 0.0:
      return float('nan')
    else:
      return self.width() / self.height()

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
