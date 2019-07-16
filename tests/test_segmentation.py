import random

import numpy as np

import pyml
from pyml import log
from pyml import mnist
from pyml import segmentation

_logger = log.setup('info')


def _setGlobalRandomSeed(seed=316):
  random.seed(seed)
  np.random.seed(seed)


def test1():

  _setGlobalRandomSeed()
  mnist = pyml.mnist.Mnist(norm_mode=0)
  random_digit = random.randrange(10)
  _logger.info("random digit: {}".format(random_digit))
  num_instances = len(mnist.digit_indices[random_digit])
  _logger.info("count of {} digits: {}".format(random_digit, num_instances))
  random_index = random.randrange(num_instances)
  _logger.info("random index: {}".format(random_index))
  random_image_index = mnist.digit_indices[random_digit][random_index]
  _logger.info("random imgae index: {}".format(random_image_index))
  random_image = mnist.images[random_image_index].reshape((28,28))
  _logger.info("image\n{}".format(1* (random_image > 0)))
  truth = pyml.mnist.Mnist.imageToGroundTruth(random_image, random_digit)
  _logger.info("Ground truth\n{}".format(truth['ground_truth']))

  nb_pixel, nb_correct, nb_false = segmentation.countMatch(truth, truth['ground_truth'])
  _logger.info("# of object pixel {} correct {} false detet {}".format(nb_pixel, nb_correct, nb_false))



if __name__ == "__main__":
  test1()
