"""Script to combine PDFs in a folder into a single PDF."""
import glob
import os
import pdb
from typing import List

from absl import app, flags
from PyPDF2 import PdfMerger

FLAGS = flags.FLAGS

flags.DEFINE_string("cohort", "all", "cohort folder name in data folder")


def get_pdfs() -> List[str]:
    """Get all pdfs in the data/ folder."""
    if FLAGS.cohort == "healthy":
        pdfs = glob.glob(os.path.join("data", "healthy", "report_clinical.pdf"))
    elif FLAGS.cohort == "cteph":
        pdfs = glob.glob(os.path.join("data", "cteph", "report_clinical.pdf"))
    elif FLAGS.cohort == "ild":
        pdfs = glob.glob(os.path.join("data", "ild", "report_clinical.pdf"))
    elif FLAGS.cohort == "tyvaso":
        pdfs = glob.glob(os.path.join("data", "tyvaso", "report_clinical.pdf"))
    elif FLAGS.cohort == "all":
        pdfs = glob.glob("data/**/report_clinical.pdf", recursive=True)
    else:
        raise ValueError("Invalid cohort name")

    pdfs.sort()
    return pdfs


def main(argv):
    """Combine PDFs in a folder into a single PDF.

    Combines all pdfs in the data/ folder into a single pdf. Save it to tmp folder.
    This may take a while to run.
    """
    pdfs = get_pdfs()
    merger = PdfMerger()
    for file in pdfs:
        merger.append(file)
    merger.write(os.path.join("tmp", "combined.pdf"))


if __name__ == "__main__":
    """Run the script to combine PDFs in a folder into a single PDF."""
    app.run(main)
