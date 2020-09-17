# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="example-pkg-izmty",  # Replace with your own username
  version="0.0.2",
  author="Sumanas Sarma",
  author_email="insectatorious+pypi@gmail.com",
  description="A simple way to calculate retail stats",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/pypa/sampleproject",
  packages=setuptools.find_packages(),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Operating System :: OS Independent",
    "Development Status :: 3 - Alpha",
    "Topic :: Office/Business"
  ],
  python_requires='>=3.7',
)