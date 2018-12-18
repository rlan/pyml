import random

import numpy as np

import pyml
import pyml.log
import pyml.mnist
#from pyml import mnist

_logger = pyml.log.setup('debug')


def _setGlobalRandomSeed(seed=316):
  random.seed(seed)
  np.random.seed(seed)

def test1():
  _logger.info("_test1()")
  m = pyml.mnist.Mnist(norm_mode=0)
  random_digit = random.randrange(10)
  _logger.info("random digit: {}".format(random_digit))
  num_instances = len(m.digit_indices[random_digit])
  _logger.info("count of {} digits: {}".format(random_digit, num_instances))
  random_index = random.randrange(num_instances)
  _logger.info("random index: {}".format(random_index))
  random_image_index = m.digit_indices[random_digit][random_index]
  _logger.info("random imgae index: {}".format(random_image_index))
  random_image = m.images[random_image_index].reshape((28,28))
  _logger.info("image\n{}".format(1* (random_image > 0)))
  ground_truth = m.imageToGroundTruth(random_image, random_digit)
  _logger.info("Ground truth\n{}".format(ground_truth['ground_truth']))



if __name__ == "__main__":
  _setGlobalRandomSeed()
  test1()
