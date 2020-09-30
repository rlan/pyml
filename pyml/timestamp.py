import time


class TimeStamp():
  def __init__(self):
    """A timestamp class

    Time string is accurate to the seconds.
    """
    self.seconds_since_epoch = time.time()
    struct = time.localtime(self.seconds_since_epoch)
    self.time_str = time.strftime("%Y%m%d-%H%M%S", struct)

  def __str__(self):
    return self.time_str

  def elapsed(self):
    """Time elapsed in seconds since creation of this instance
    """
    return time.time() - self.seconds_since_epoch


