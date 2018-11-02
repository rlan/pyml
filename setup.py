import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

exec(open('pyml/version.py').read())

setuptools.setup(
  name="pyml",
  version=__version__,
  author="Rick Lan",
  author_email="rlan@users.noreply.github.com",
  description="A Machine Learning Utility Library",
  long_description=long_description,
  long_description_content_type="text/markdown",
  license="MIT License",
  url="https://github.com/rlan/pyml",
  packages=setuptools.find_packages(),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  install_requires=[
    "numpy >= 1.13.3",
    "sklearn",
  ]
)
