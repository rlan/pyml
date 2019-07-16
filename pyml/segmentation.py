import random

import numpy as np

from . import log
_logger = log.setup('info')


def countMatch(truth, candidate):
  """Count matches

  Parameters
  ----------
  truth : dict
      'digit' : int
      'ground_truth' : numpy.ndarry
      'mask' : numpy.ndarray
  candiate : numpy.ndarray
  
  Returns
  -------
  tuple
      int : Total number of pixel of the object.
      int : Total number of correct object pixel.
      int : Total number of false detected object pixel.
  """
  digit = truth['digit']
  ground_truth = truth['ground_truth']
  object_mask = truth['mask']
  nb_obj_pixel = np.sum(object_mask)
  nb_correct_pixel = np.sum(ground_truth[object_mask] == candidate[object_mask])
  
  digit_matrix = np.full(ground_truth.shape, digit+1, ground_truth.dtype)
  inverse_object_mask = np.logical_not(object_mask)
  nb_false_pixel = np.sum(candidate[inverse_object_mask] == digit_matrix[inverse_object_mask])

  #np_miss_pixel = np_obj_pixel - nb_correct_pixel

  return nb_obj_pixel, nb_correct_pixel, nb_false_pixel
