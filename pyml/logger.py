"""Global logger

Example
-------

from pyml import logger
logger.info('hello')
logger.debug('world')
"""

import logging

default_level = logging.INFO
default_formatting = '%(levelname)s:%(filename)s:%(lineno)d:%(message)s'

logger = logging.getLogger()
logger.setLevel(default_level)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(default_level)
# create formatter and add it to the handlers
formatter = logging.Formatter(default_formatting)
ch.setFormatter(formatter)
logger.addHandler(ch)


def critical(*args):
  logger.critical(*args)

def error(*args):
  logger.error(*args)

def warrning(*args):
  logger.warning(*args)

def info(*args):
  logger.info(*args)

def debug(*args):
  logger.debug(*args)

def setLevel(level='info'):
  """Setup log level filter
  
  Parameters
  ----------
  log_level : str
      One of the following: 'critical', 'error', 'warning', 'info' and 'debug'.
      Default is 'info'.
  """

  mapper = {
    'critical' : logging.CRITICAL, 
    'error' : logging.ERROR,
    'warning' : logging.WARNING,
    'info' : logging.INFO,
    'debug' : logging.DEBUG,
  }
  if level not in mapper:
    raise ValueError('level must be one of these: {}'.format(list(mapper.keys())))
  else:
    logger.setLevel(mapper[level])

