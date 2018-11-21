import logging

def setup(log_level='info'):
  """Setup log level filter
  
  Parameters
  ----------
  log_level : str
      One of the following: 'critical', 'error', 'warning', 'info', 'debug',
      'notset'. Default is 'info'.
  """

  mapper = {  'critical' : logging.CRITICAL, 
              'error' : logging.ERROR,
              'warning' : logging.WARNING,
              'info' : logging.INFO,
              'debug' : logging.DEBUG,
              'notset' : logging.NOTSET }
  if log_level not in mapper:
    raise ValueError('Invalid log level given: {}'.format(log_level))
  this_level = mapper[log_level]    
  #logger = logging.getLogger('app')
  logger = logging.getLogger()
  logger.setLevel(this_level)
  # create console handler with a higher log level
  ch = logging.StreamHandler()
  ch.setLevel(this_level)
  # create formatter and add it to the handlers
  formatter = logging.Formatter('%(levelname)s:%(filename)s:%(lineno)d:%(message)s')
  ch.setFormatter(formatter)
  logger.addHandler(ch)
  return logger
