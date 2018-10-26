"""
Image related routines
"""
from __future__ import division
from __future__ import print_function

import numpy as np
from skimage import io

def floatToUint8(img, scale = None, minmax = None, scretch = False):
  """Convert floating-point matrix to 8-bit matrix for writing as JPG file

  Parameters
  ----------
  img : ndarray
      Input matrix
  scale : float
      If None, no scaling is applied.
  minmax : tuple of float
      If None, no scaling is applied. Otherwise input is scaled such that it fits
      in the result [min, max].
  scretch : boolean
      If true, scretch min and max of input signal to occupy the 8-bit output.

  Returns
  -------
  ndarray
      Image
  """
  #print("shape", img.shape)

  if scale != None:
    img = img * scale

  if minmax != None:
    amax = np.amax(img)
    amin = np.amin(img)
    scale = (minmax[1]-minmax[0]) / (amax - amin)
    img = (img-amin) * scale

  if scretch:
    amax = np.amax(img)
    amin = np.amin(img)
    #print("amax", amax, "amin", amin)
    if amax != amin:
      scale = 255.0/(amax-amin)
      img = (img-amin) * scale

  img = np.clip(img, 0.0, 255.0)
  return img.astype(dtype='uint8')

def saveImage(image, shape=(28, 28), file_name=None):
  """Save image array as jpeg file.

  Parameters
  ----------
  image : numpy.ndarray

  Returns
  -------
  None
  """
  img = image.reshape(shape)
  if file_name == None:
      io.imsave('_image.jpg', img)
  else:
      io.imsave(file_name, img)


if __name__ == "__main__":
  # TODO need tests
  pass
