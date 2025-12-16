from math import sqrt

import numpy as np
from numpy import mean
from numpy import var
from scipy.stats import wilcoxon

import glob
import os
import pandas as pd


def cohend(d1, d2):
    """
    function to calculate Cohen's d for independent samples
    """

    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)
    # calculate the effect size
    d = (u1 - u2) / s
    d = abs(d)

    result = ""
    if d < 0.2:
        result = "negligible"
    if 0.2 <= d < 0.5:
        result = "small"
    if 0.5 <= d < 0.8:
        result = "medium"
    if d >= 0.8:
        result = "large"

    return result, d


def run_wilcoxon_and_cohend(data1, data2):
    res = wilcoxon(
        x=data1,
        y=data2,
        zero_method="wilcox",
        alternative="two-sided",
        # mode='a',
        method="auto",  # len<50 approx else exact
    )
    cohensd = cohend(data1, data2)
    print(f"statistic is: {res.statistic} P-Value is: {res.pvalue}")
    print(f"Cohen's D is: {cohensd}")

    return res.pvalue, cohensd


def main():
    name_list = glob.glob(os.path.join("../result", "csv_folder", "record_*.csv"))
    name_list.sort(key=str.lower)

    # change order of r_r
    to_be_moved = [i for i in name_list if "R_R" in i]
    name_list.remove(to_be_moved[0])
    name_list.append(to_be_moved[0])

    print(name_list)

    col_names = [("_").join(name.split("/")[-1][:-4].split("_")[1:]) for name in name_list]

    df = pd.DataFrame(columns=col_names, index=["Pvalue", "CohenD"])
    df.idx = np.linspace(1, 1000, 1000)

    df_random = pd.read_csv("../result/csv_folder/record_R_R.csv")
    df_random["id"] = df_random["id"].astype(str) + "_" + df_random["expected_label"].astype(str)
    df_random = df_random[["method", "id", "mutation_number"]]

    # for col in col_name:
    for col, csv_file in zip(col_names, name_list):
        if col != "R_R":
            df_xai = pd.read_csv(csv_file)
            df_xai["id"] = df_xai["id"].astype(str) + "_" + df_xai["expected_label"].astype(str)
            df_xai = df_xai[["method", "id", "mutation_number"]]
            df_merge = pd.merge(df_random, df_xai, on="id")

            print("---------------------------------------------------------")
            print(f"Test for {col} and random")
            pvalue, cohensd = run_wilcoxon_and_cohend(
                df_merge["mutation_number_x"], df_merge["mutation_number_y"]
            )
            df.at["Pvalue", col] = pvalue
            df.at["CohenD", col] = cohensd[0]

    print(df)
    df.to_csv("../result/csv_folder/stat_tests.csv")


if __name__ == "__main__":
    main()
