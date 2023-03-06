"""Output statistics for the paper."""
# import modules
import pdb
import sys

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from absl import app, flags

FILE_CSV = "data/stats_paper_all.csv"


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Label all subject IDs with a prefix of "005" as "CTEPH", a prefix of "005"
    and ending in "A" as "post-CTEPH", and all other subject IDs as "healthy".

    Args:
        path (str): path to CSV file

    Returns:
        pandas DataFrame
    """
    df = pd.read_csv(path)
    df["subject_id"] = df["subject_id"].astype(str)
    df["group"] = df["subject_id"].apply(
        lambda x: "CTEPH-post"
        if x.startswith("005") and "A" in x
        else ("CTEPH-pre" if x.startswith("005") and "A" not in x else "healthy")
    )
    return df


def analyze_oscillation_defect(df: pd.DataFrame) -> None:
    """Analyze the oscillation defect + low data."""
    # combine the two CTEPH groups and perform t-test with healthy
    cteph_df = df
    # perform mann-whitney u test
    res = stats.mannwhitneyu(
        cteph_df[cteph_df["group"] == "healthy"]["osc_defectlow"],
        cteph_df[cteph_df["group"] == "CTEPH-pre"]["osc_defectlow"],
    )
    # print the p-values
    print(f"oscillation defect p-value: {res.pvalue}")
    # plot the data
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.figure()
    ax = sns.boxplot(x="group", y="osc_defect", data=cteph_df, palette="Set2")
    ax.set_title("Oscillation defect + Low")
    ax.set_ylabel("Oscillation defect + Low %")
    plt.savefig("tmp/stats_paper_osc_defect.png")
    cteph_df.to_csv("data/stats_paper_between.csv", index=False)


def analyze_oscillation_mean(df: pd.DataFrame) -> None:
    """Analyze the oscillation mean data."""
    cteph_df = df
    # perform mann whitney u test
    res = stats.mannwhitneyu(
        cteph_df[cteph_df["group"] == "healthy"]["osc_mean"],
        cteph_df[cteph_df["group"] == "CTEPH-pre"]["osc_mean"],
    )
    # print the p-values
    print(f"oscillation defect p-value: {res.pvalue}")
    # plot the data
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.figure()
    ax = sns.boxplot(x="group", y="osc_mean", data=cteph_df, palette="Set2")
    ax.set_title("Oscillation Mean")
    ax.set_ylabel("Oscillation Mean %")
    plt.savefig("tmp/stats_paper_osc_mean.png")


def analyze_paired_oscillation_defect(df: pd.DataFrame) -> None:
    """Analyze the paired oscillation data.

    Only use the data of CTEPH_pre subjects that have a corresponding CTEPH_post.
    """
    # remove healthy subjects
    cteph_df = df[df["group"] != "healthy"]
    # filter out CTEPH_pre subjects that don't have a corresponding CTEPH_post
    cteph_pre_df = cteph_df[cteph_df["group"] == "CTEPH-pre"]
    cteph_pre_df = cteph_pre_df[
        cteph_pre_df["subject_id"].apply(lambda x: x + "A" in list(df["subject_id"]))
    ]
    # filter out CTEPH_post subjects that don't have a corresponding CTEPH_pre
    cteph_post_df = cteph_df[cteph_df["group"] == "CTEPH-post"]
    # combine the CTEPH_pre and CTEPH_post data
    cteph_df = pd.concat([cteph_pre_df, cteph_post_df])
    # perform t test
    res = stats.wilcoxon(
        cteph_df[cteph_df["group"] == "CTEPH-pre"]["osc_defect"],
        cteph_df[cteph_df["group"] == "CTEPH-post"]["osc_defect"],
    )
    # print the p-values
    print(f"oscillation defect p-value: {res.pvalue}")
    # plot the data
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.figure()
    ax = sns.boxplot(x="group", y="osc_defectlow", data=cteph_df, palette="Set2")
    ax.set_title("Oscillation defect difference")
    ax.set_ylabel("Oscillation defect difference %")
    # add p-value to plot
    x1, x2 = (
        0,
        1,
    )  # columns 'CTEPH-pre' and 'CTEPH-post' (first column: 0, see plt.xticks())
    y, h, col = cteph_df["osc_defectlow"].max() + 0.2, 0.2, "k"
    ax.text(
        (x1 + x2) * 0.5,
        y + h,
        f"p = {res.pvalue:.3f}",
        ha="center",
        va="bottom",
        color=col,
    )
    plt.savefig("tmp/stats_paper_osc_defect_diff.png")
    # export dataframe
    cteph_df.to_csv("data/stats_paper_paired.csv")


def analyze_paired_oscillation_mean(df: pd.DataFrame) -> None:
    """Analyze the paired oscillation data.

    Only use the data of CTEPH_pre subjects that have a corresponding CTEPH_post.
    """
    # remove healthy subjects
    cteph_df = df[df["group"] != "healthy"]
    # filter out CTEPH_pre subjects that don't have a corresponding CTEPH_post
    cteph_pre_df = cteph_df[cteph_df["group"] == "CTEPH-pre"]
    cteph_pre_df = cteph_pre_df[
        cteph_pre_df["subject_id"].apply(lambda x: x + "A" in list(df["subject_id"]))
    ]
    # filter out CTEPH_post subjects that don't have a corresponding CTEPH_pre
    cteph_post_df = cteph_df[cteph_df["group"] == "CTEPH-post"]
    # combine the CTEPH_pre and CTEPH_post data
    cteph_df = pd.concat([cteph_pre_df, cteph_post_df])
    # perform t test
    res = stats.wilcoxon(
        cteph_df[cteph_df["group"] == "CTEPH-pre"]["osc_mean"],
        cteph_df[cteph_df["group"] == "CTEPH-post"]["osc_mean"],
    )
    # print the p-values
    print(f"oscillation defect p-value: {res.pvalue}")
    # plot the data
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.figure()
    ax = sns.boxplot(x="group", y="osc_mean", data=cteph_df, palette="Set2")
    ax.set_title("Oscillation mean")
    ax.set_ylabel("Oscillation mean %")
    # add p-value to plot
    x1, x2 = (
        0,
        1,
    )  # columns 'CTEPH-pre' and 'CTEPH-post' (first column: 0, see plt.xticks())
    y, h, col = cteph_df["osc_mean"].max() + 0.2, 0.2, "k"
    ax.text(
        (x1 + x2) * 0.5,
        y + h,
        f"p = {res.pvalue:.3f}",
        ha="center",
        va="bottom",
        color=col,
    )
    plt.savefig("tmp/stats_paper_osc_mean_diff.png")


def main(unused_argv):
    """Perform t tests on the data."""
    df = load_csv("data/stats_paper_all.csv")
    analyze_oscillation_defect(df)
    analyze_oscillation_mean(df)
    analyze_paired_oscillation_defect(df)
    analyze_paired_oscillation_mean(df)


if __name__ == "__main__":
    app.run(main)
