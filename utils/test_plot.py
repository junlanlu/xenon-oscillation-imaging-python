"""Test plot.py."""

import pdb

import constants
import numpy as np
import plot
from absl import app


def test_plot_montage_grey():
    """Test plot_montage_grey."""
    image = np.random.rand(128, 128, 16)
    plot.plot_montage_grey(
        image=image, path="../tmp/montage.png", index_start=0, index_skip=1
    )


def test_plot_montage_color():
    """Test plot_montage_color."""
    image = (7 * np.random.rand(128, 128, 16)).astype(int)
    image = plot.map_grey_to_rgb(image, cmap=constants.CMAP.RBC_OSC_BIN2COLOR)
    plot.plot_montage_color(
        image=image, path="../tmp/montage.png", index_start=0, index_skip=1
    )


def test_plot_histogram_rbc_osc():
    """Test plot_rbc_osc_histogram."""
    data = 30 * np.random.normal(0, 0.1, 10000)
    data[data < -20] = -20
    data[data > 30] = 30
    plot.plot_histogram_rbc_osc(data=data, path="../tmp/histogram.png")


def main(argv):
    """Run tests."""
    test_plot_histogram_rbc_osc()


if __name__ == "__main__":
    app.run(main)
