"""Test io_utils.py.

User must specify the path to the twix file.
"""
import logging
import pdb

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import signal_utils
from absl import app


def test_get_heart_rate():
    """Test get heart rate.

    Heart rate should be 60bpm.
    """
    t = np.arange(0, 10, 0.1)
    data = np.sin(2 * np.pi * t)
    heart_rate = signal_utils.get_heartrate(data, 0.1)
    logging.info("Heart rate: %f BPM", heart_rate)


def test_find_high_low_indices():
    """Test find high low indices."""
    step = 0.01
    t = np.arange(0, 10, step)
    data = np.sin(2 * np.pi * t)
    heart_rate = signal_utils.get_heartrate(data, step)
    peak_distance = int((heart_rate / 60) / step) // 2
    high_indices, low_indices = signal_utils.find_high_low_indices(data, peak_distance)
    plt.figure()
    plt.plot(t, data, "k.")
    plt.plot(t[high_indices], data[high_indices], "r.")
    plt.plot(t[low_indices], data[low_indices], "b.")
    plt.show()


def main(argv):
    """Run tests."""
    test_get_heart_rate()
    # test_find_high_low_indices()


if __name__ == "__main__":
    app.run(main)
