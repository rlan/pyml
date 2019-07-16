from __future__ import division
from __future__ import print_function

from datetime import datetime
import logging
import os
import random
import sys

import numpy as np
from sklearn.datasets import fetch_mldata
from . import log

# Attach to global logger
_logger = log.setup('info')

class Mnist:
  """Fetch, store and serve MNIST dataset.

  Parameters
  ----------
  data_home : str
      Location of local storage for Mnist dataset.
  norm_mode : int
      Image normalization mode for Mnist images.
      0 (default) - raw values.
      1 - normalize to [0, 1]
      2 - normalize to [-1, 1]

  Returns
  -------
  None
  """

  def __init__(self, data_home = '~/.pyml/mnist', norm_mode=0):
    """Constructor
    """
    _logger.info("Saving MNIST dataset in {} ...".format(data_home))
    t0 = datetime.now()
    dataset = fetch_mldata('MNIST original', data_home=data_home)
    t1 = datetime.now()
    tdelta = t1 - t0
    _logger.info("Dataset saved in {} seconds".format(tdelta.total_seconds()))
    self._inspectDataset(dataset)

    self.digit_indices = dict()
    for digit in range(0, 10):
      self.digit_indices[digit] = np.flatnonzero(dataset.target == digit)
    self.images = self.normalize(dataset.data, norm_mode)
    self._inspectDatasetStats(self.digit_indices, dataset.data)
    self._inspectImages(dataset.data)
    self._inspectImages(self.images)

    # Split into train and test set
    self.train_digit_indices = dict()
    self.test_digit_indices = dict()
    for digit in range(0, 10):
      stop = round(self.digit_indices[digit].size * 6.0/7.0)
      self.train_digit_indices[digit] = self.digit_indices[digit][:stop]
      self.test_digit_indices[digit] = self.digit_indices[digit][stop:]
    _logger.debug("train_digit_indices")
    self._inspectDatasetStats(self.train_digit_indices, dataset.data)
    _logger.debug("test_digit_indices")
    self._inspectDatasetStats(self.test_digit_indices, dataset.data)

    tdelta = datetime.now() - t1
    _logger.info("Dataset processed in {} seconds".format(tdelta.total_seconds()))

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
  def normalize(images, norm_mode=0):
    """Normalize image data

    Parameters
    ----------
    images : numpy.ndarray

    norm_mode : int
        0 (default) - raw values.
        1 - normalize to [0, 1]
        2 - normalize to [-1, 1]

    Returns
    -------
    numpy.ndarray(dtype='uint8') or numpy.ndarray(dtype='float32')
        Raw or normalized images.
    """
    if norm_mode == 0:
      return images

    # normalize to [0, 1]
    _logger.debug("images.dtype {} should be uint8".format(images.dtype))
    images = images.astype('float32') / 255.0
    _logger.debug("images.dtype {} should be float".format(images.dtype))

    # normalize to [-1, 1]
    # class torchvision.transforms.Normalize(mean, std)
    # input[channel] = (input[channel] - mean[channel]) / std[channel]
    if norm_mode > 1:
      # MNIST (mean, std) = (0.1307,), (0.3081,)
      images = (images -  0.1307) / 0.3081

    _logger.debug("images.dtype {} should be float".format(images.dtype))
    return images

  @staticmethod
  def unnormalize(images, norm_mode=0):
    """Un-normalize MNIST data
    Un-normalize from [0, 1] (default) or from [-1, 1].

    Parameters
    ----------
    images : numpy.ndarray

    norm_mode : bool
        If True, un-normalize from [-1, 1]. If false, just default un-normalization from [0, 1].
        0 (default) - raw values.
        1 - normalize to [0, 1]
        2 - normalize to [-1, 1]

    Returns
    -------
    numpy.ndarray(dtype='uint8')
        Un-normalized
    """
    if norm_mode == 0:
      return images

    if norm_mode > 1:
      # MNIST (mean, std) = (0.1307,), (0.3081,)
      images = images*0.3081 + 0.1307

    images = images * 255.0
    np.clip(images, 0.0, 255.0)
    images = images.astype('uint8')
    return images


  @staticmethod
  def imageToGroundTruth(image, digit):
    """Convert image to ground truth

    Example image:
      A 'one':
    . . . . .
    . . X . .
    . . X . .
    . . X . .
    . . . . .

    Example Ground truth:
    0 0 0 0 0
    0 0 2 0 0
    0 0 2 0 0
    0 0 2 0 0
    0 0 0 0 0

    Parameters
    ----------
    image : numpy.ndarray
        Array representing an image.
    digit : int
        Digit in the image.

    Returns
    -------
    dict
        'digit' : int
            digit of the image
        'ground_truth' : numpy.ndarray(shape=image.shape, dtype=image.dtype)
            Grouth truth. See an example above.
    """
    ret = np.full(image.shape, 0, dtype=image.dtype)
    mask = image > 0
    ret[mask] = digit+1
    return {'digit': digit, 'ground_truth' : ret, 'mask' : mask}
