"""Test binning.py"""

import pdb

import binning
import numpy as np
from absl import app


def test_linear_bin():
    image = np.random.rand(10, 10)
    mask = np.ones((10, 10))
    thresholds = np.array([0.25, 0.5, 0.75])
    image_binned = binning.linear_bin(image, mask, thresholds)


def main(argv):
    test_linear_bin()


if __name__ == "__main__":
    app.run(main)
