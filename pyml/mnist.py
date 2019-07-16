from __future__ import division
from __future__ import print_function

from datetime import datetime
import logging
import os
import os.path
import pickle
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
  self.images : An array of vector representing the pixel value of the image.
  self.digit_indices : A 2-D array whose first axis is the digit and second axis
      the index to self.images.
  self.train_digit_indices : Submatrix of self.digit_indices for training.
  self.test_digit_indices : Submatrix of self.digit_indices for test.
  """

  def __init__(self, data_home = os.path.expanduser('~') +'/.pyml/mnist', norm_mode=0):
    """Constructor
    """
    if norm_mode < 0 or norm_mode > 2:
      raise ValueError("Invalid norm_mode: {}".format(norm_mode))

    file_name = data_home + "/data{}.pkl".format(norm_mode)
    if os.path.isfile(file_name):
      _logger.info("Loading MNIST dataset from {} ...".format(file_name))
      t0 = datetime.now()
      self._data = pickle.load(open(file_name, 'rb'))
      tdelta = datetime.now() - t0
      _logger.info("Loaded in {} seconds".format(tdelta.total_seconds()))
    else:
      dataset = Mnist._fetchMNIST(data_home)
      self._data = Mnist._processMNIST(dataset, norm_mode)
      _logger.info("Saving post-processed data to {} ...".format(file_name))
      t0 = datetime.now()
      pickle.dump(self._data, open(file_name, 'wb'))
      tdelta = datetime.now() - t0
      _logger.info("Saved in {} seconds".format(tdelta.total_seconds()))

    # Unpack data
    self.images = self._data['images']
    self.digit_indices = self._data['digit_indices']
    self.train_digit_indices = self._data['train_digit_indices']
    self.test_digit_indices = self._data['test_digit_indices']


  @staticmethod
  def _fetchMNIST(data_home = os.path.expanduser('~')+'/.pyml/mnist'):
    _logger.info("Fetching MNIST dataset in {} ...".format(data_home))
    t0 = datetime.now()
    dataset = fetch_mldata('MNIST original', data_home=data_home)
    Mnist._inspectDataset(dataset)
    t1 = datetime.now()
    tdelta = t1 - t0
    _logger.info("Fetched in {} seconds".format(tdelta.total_seconds()))
    return dataset

  
  @staticmethod
  def _processMNIST(dataset, norm_mode=0):
    _logger.info("Post-processing MNIST dataset...")
    t0 = datetime.now()
    digit_indices = dict()
    for digit in range(0, 10):
      digit_indices[digit] = np.flatnonzero(dataset.target == digit)
    images = Mnist.normalize(dataset.data, norm_mode)
    Mnist._inspectDatasetStats(digit_indices, dataset.data)
    Mnist._inspectImages(dataset.data)
    Mnist._inspectImages(images)

    # Split into train and test set
    train_digit_indices = dict()
    test_digit_indices = dict()
    for digit in range(0, 10):
      stop = round(digit_indices[digit].size * 6.0/7.0)
      train_digit_indices[digit] = digit_indices[digit][:stop]
      test_digit_indices[digit] = digit_indices[digit][stop:]
    _logger.debug("train_digit_indices")
    Mnist._inspectDatasetStats(train_digit_indices, dataset.data)
    _logger.debug("test_digit_indices")
    Mnist._inspectDatasetStats(test_digit_indices, dataset.data)

    tdelta = datetime.now() - t0
    _logger.info("Post-processed in {} seconds".format(tdelta.total_seconds()))

    ret = { 
      'images' : images, 
      'digit_indices' : digit_indices, 
      'train_digit_indices' : train_digit_indices, 
      'test_digit_indices' : test_digit_indices
    }
    return ret


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
