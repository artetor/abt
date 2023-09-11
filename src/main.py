import numpy as np
from scipy import stats
import pandas as pd


def find_means(old_vec: np.ndarray, new_vec: np.ndarray):
    return np.mean(old_vec), np.mean(new_vec)


def find_percentiles(old_vec: np.ndarray, new_vec: np.ndarray, percentile: float):
    return np.percentile(old_vec, percentile), np.percentile(new_vec, percentile)


def t_test(old_vec: np.ndarray, new_vec: np.ndarray):
    statistic, p_value = stats.ttest_ind(old_vec, new_vec)
    return statistic, p_value


def anova(old_vec: np.ndarray, new_vec: np.ndarray):
    statistic, p_value = stats.f_oneway(old_vec, new_vec)
    return statistic, p_value


def reduce_uids(df: pd.DataFrame):
    return df.groupby("uid").agg(max)
