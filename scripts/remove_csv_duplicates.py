"""Remove duplicate rows from a CSV file."""

import pandas as pd
from absl import app, flags

CSV_FILE = "data/stats_all.csv"


def remove_duplicates(path: str):
    """Remove duplicate rows from a CSV file.

    Checks for duplicate subject IDs and removes all the extra rows besides the
    most recent one.

    Also sort the rows by subject ID.

    Args:
        path (str): path to CSV file
    """
    df = pd.read_csv(path)
    df.drop_duplicates(subset="subject_id", keep="last", inplace=True)
    df.sort_values(by="subject_id", inplace=True)
    df.to_csv(path, index=False)


def main(unused_argv):
    """Main function to remove duplicates from a CSV file."""
    remove_duplicates(CSV_FILE)


if __name__ == "__main__":
    app.run(main)
