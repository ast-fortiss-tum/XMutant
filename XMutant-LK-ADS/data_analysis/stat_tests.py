from math import sqrt

import numpy as np
from numpy import mean
from numpy import var
from scipy.stats import wilcoxon


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

    result = ''
    if d < 0.2:
        result = 'negligible'
    if 0.2 <= d < 0.5:
        result = 'small'
    if 0.5 <= d < 0.8:
        result = 'medium'
    if d >= 0.8:
        result = 'large'

    return result, d


def run_wilcoxon_and_cohend(data1, data2):
    res = wilcoxon(x=data1, y=data2,
                   zero_method = 'wilcox',
                   alternative = "two-sided",
                   #mode='a',
                   method='auto'# len<50 approx else exact
                   )
    cohensd = cohend(data1, data2)
    print(f"statistic is: {res.statistic} P-Value is: {res.pvalue}")
    print(f"Cohen's D is: {cohensd}")

    return res.pvalue, cohensd

def run_wilcoxon(data1):
    print("Length of data ", len(data1))
    res = wilcoxon(x=data1,
                   zero_method = 'wilcox',
               alternative = "two-sided",
                   #mode='a',
                   method='auto'# len<50 approx else exact
                   )
    #cohensd = cohend(data1, data2)
    print(f"statistic is: {res.statistic} P-Value is: {res.pvalue}")
    #print(f"Cohen's D is: {cohensd}")

    return res.pvalue#, cohensd

def main():
    # data1 =  # first distribution
    # data2 =  # second distribution

    run_wilcoxon_and_cohend(data1, data2)


if __name__ == '__main__':
    main()
