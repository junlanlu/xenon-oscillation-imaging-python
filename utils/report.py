"""Make reports."""
import os
import pdb
from typing import Any, Dict


def clinical(stats_dict: Dict[str, Any], path: str):
    """Make clinical report."""
    current_path = os.path.dirname(__file__)
    path_clinical = os.path.abspath(
        os.path.join(current_path, os.pardir, "assets", "html", "clinical.html")
    )
    with open(path_clinical, "r") as f:
        file = f.read()
        rendered = file.format(**stats_dict)
    with open(path, "w") as o:
        o.write(rendered)
