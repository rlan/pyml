from __future__ import division
from __future__ import print_function

from datetime import datetime
import logging
import os
import random
import sys

import numpy as np
from sklearn.datasets import fetch_mldata


# Attach to global logger
_logger = logging.getLogger('app.' + os.path.basename(__file__))

class MNIST:
  """Fetch, store and serve MNIST dataset.
  """

  def __init__(self, data_home = './datasets/mnist', norm_mode=False):
    _logger.info("Saving MNIST data in {} ...".format(data_home))
    t0 = datetime.now()
    dataset = fetch_mldata('MNIST original', data_home=data_home)
    tdelta = datetime.now() - t0
    _logger.info("Data saved in {} seconds".format(tdelta.total_seconds()))
    self._inspectDataset(dataset)

    self.digit_indices = dict()
    for digit in range(0, 10):
      self.digit_indices[digit] = np.flatnonzero(dataset.target == digit)
    self.images = self.normalize(dataset.data, norm_mode)
    self._inspectDatasetStats(self.digit_indices, dataset.data)
    self._inspectImages(dataset.data)
    self._inspectImages(self.images)

  @staticmethod
  def _inspectDataset(dataset):
    # Inspection
    # dataset.data contains images. Each image is in a row.
    # dataset.target contains labels.
    _logger.debug("dataset.data.shape {}".format(dataset.data.shape))
    _logger.debug("type(dataset.data.shape) {}".format(type(dataset.data.shape)))
    _logger.debug("dataset.target.shape {}".format(dataset.target.shape))
    _logger.debug("type(dataset.target.shape) {}".format(type(dataset.target.shape)))

  @staticmethod
  def _inspectDatasetStats(digit_indices, images):
    # Show stats of the dataset
    _logger.debug("type(digit_indices) %s", str(type(digit_indices)))
    _logger.debug("digit_indices[0].shape %s", str(digit_indices[0].shape))
    _logger.debug("images.shape %s", str(images.shape))
    _logger.debug("type(images) %s", str(type(images)))
    for i in range(0, 10):
      _logger.debug("{} has {} data point(s)".format(i, len(digit_indices[i])))
    
  @staticmethod
  def _inspectImages(images):
    _logger.debug("images.min {} .max {}".format(np.amin(images), np.amax(images)))
    _logger.debug("images.mean {} .std {}".format(np.mean(images), np.std(images)))


  @staticmethod
  def normalize(images, norm_mode):
    """Normalize MNIST data
    Normalize to [0, 1] by default and optionally to [-1, 1].

    Parameters
    ----------
    images : numpy.ndarray

    norm_mode : bool
        If True, normalize to [-1, 1]. If false, just default normalization of [0, 1].

    Returns
    -------
    numpy.ndarray(dtype='float32')
        Normalized
    """
    # normalize to [0, 1]
    _logger.debug("images.dtype {} should be uint8".format(images.dtype))
    images = images.astype('float32') / 255.0
    _logger.debug("images.dtype {} should be float".format(images.dtype))

    # class torchvision.transforms.Normalize(mean, std)
    # input[channel] = (input[channel] - mean[channel]) / std[channel]
    if norm_mode:
      # MNIST (mean, std) = (0.1307,), (0.3081,)
      images = (images -  0.1307) / 0.3081

    _logger.debug("images.dtype {} should be float".format(images.dtype))
    return images

  @staticmethod
  def unnormalize(images, norm_mode):
    """Un-normalize MNIST data
    Un-normalize from [0, 1] (default) or from [-1, 1].

    Parameters
    ----------
    images : numpy.ndarray

    norm_mode : bool
        If True, un-normalize from [-1, 1]. If false, just default un-normalization from [0, 1].

    Returns
    -------
    numpy.ndarray(dtype='uint8')
        Un-normalized
    """
    if norm_mode:
      # MNIST (mean, std) = (0.1307,), (0.3081,)
      images = images*0.3081 + 0.1307

    images = images * 255.0
    np.clip(images, 0.0, 255.0)
    images = images.astype('uint8')
    return images



def _setGlobalRandomSeed(seed=316):
  random.seed(seed)
  np.random.seed(seed)

def _setupLogging():
  logger = logging.getLogger('app')
  logger.setLevel(logging.DEBUG)
  # create console handler with a higher log level
  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  # create formatter and add it to the handlers
  formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
  ch.setFormatter(formatter)
  logger.addHandler(ch)
  return logger

def _test1():
  mnist = MNIST(norm_mode=True)
  random_digit = random.randrange(10)
  _logger.info("random digit: {}".format(random_digit))
  num_instances = len(mnist.digit_indices[random_digit])
  _logger.info("count of {} digits: {}".format(random_digit, num_instances))
  random_index = random.randrange(num_instances)
  _logger.info("random index: {}".format(random_index))
  _logger.info("image\n{}".format(1* (mnist.images[random_index].reshape((28,28)) > 0)))



if __name__ == "__main__":
  _setGlobalRandomSeed()
  _setupLogging()
  _test1()