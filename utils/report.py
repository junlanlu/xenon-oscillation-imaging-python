"""Make reports."""
import os
import pdb
import sys
from typing import Any, Dict

import pdfkit

PDF_OPTIONS = {
    "page-width": 300,
    "page-height": 150,
    "margin-top": 1,
    "margin-right": 0.1,
    "margin-bottom": 0.1,
    "margin-left": 0.1,
    "dpi": 300,
    "encoding": "UTF-8",
    "enable-local-file-access": None,
}


def format_dict(stats_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Format dictionary for report.

    Rounds values to 2 decimal places.
    Args:
        stats_dict (Dict[str, Any]): dictionary of statistics
    Returns:
        Dict[str, Any]: formatted dictionary
    """
    for key in stats_dict.keys():
        if isinstance(stats_dict[key], float):
            stats_dict[key] = round(stats_dict[key], 2)
    return stats_dict


def clinical(stats_dict: Dict[str, Any], path: str):
    """Make clinical report.

    First converts dictionary to html format. Then saves to path.
    Args:
        stats_dict (Dict[str, Any]): dictionary of statistics
        path (str): path to save report
    """
    stats_dict = format_dict(stats_dict)
    current_path = os.path.dirname(__file__)
    path_clinical = os.path.abspath(
        os.path.join(current_path, os.pardir, "assets", "html", "clinical.html")
    )
    path_html = os.path.join("tmp", "clinical.html")
    # write report to html
    with open(path_clinical, "r") as f:
        file = f.read()
        rendered = file.format(**stats_dict)
    with open(path_html, "w") as o:
        o.write(rendered)
    # write clinical report to pdf
    pdfkit.from_file(path_html, path, options=PDF_OPTIONS)
