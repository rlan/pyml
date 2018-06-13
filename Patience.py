class Patience:
  """Patience tracker

  If calmness is not increased at each test, tolerance is decreased. When
  tolerance reaches 0, the call to object return False, e.g. no more patience.

  If calmness is increased at each test, tolerance is reset to the preset value.

  When tolerance is > 0, the call to object returns True, e.g, has patience.

  Example:
  An usage of this is to keep training the network until X number of epochs does
  not return a better accuracy, at which point the training is stopped.

  >>> from Patience import Patience
  >>> patience = Patience(3)
  >>> patience()
  True

  Default calmnness is 0. 0.1 is higher.
  >>> _ = patience.test(0.1)
  >>> patience()
  True

  0.1 is same, but not better before. Tolerance starts count down.
  >>> _ = patience.test(0.1)
  >>> patience()
  True
  >>> _ = patience.test(0.1)
  >>> patience()
  True
  >>> _ = patience.test(0.1)

  After 3rd test, tolerance runs out.
  >>> patience()
  False
  """

  def __init__(self, capacity = 10):
    self._tolerance = self._capacity = int(capacity)
    self._calmness = 0.0
  
  def __call__(self):
    if self._tolerance > 0:
      return True
    else:
      return False

  def test(self, calmness):
    if calmness > self._calmness:
      self._calmness = calmness
      self._tolerance = self._capacity
    else:
      self._tolerance -= 1
    return self


if __name__ == "__main__":
  import doctest 
  import sys 
  (failure_count, test_count) = doctest.testmod() 
  sys.exit(failure_count) 
