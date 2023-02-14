"""Test report.py"""

import pdb

from absl import app

from utils import report


def test_clinical():
    """Test clinical report."""
    stats_dict = {
        "subject_id": "test",
        "scan_date": "test",
        "process_date": "test",
    }
    report.clinical(stats_dict, "tmp/test_clinical.html")


def main(argv):
    """Run tests."""
    test_clinical()


if __name__ == "__main__":
    app.run(main)
