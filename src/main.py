from typing import Union, List
import numpy as np
from scipy import stats
import pandas as pd


def find_means(df: pd.DataFrame):
    means = df.mean()

    if ("conv" in df.columns) and ("revenue" in df.columns):
        means["revenue_nonzeros"] = df[df["conv"] != 0]["revenue"].mean()

    return means


def find_percentiles(df: pd.DataFrame, percentile: float):
    percentiles = df.apply(lambda col: np.percentile(col, percentile))

    if ("conv" in df.columns) and ("revenue" in df.columns):
        percentiles["revenue_nonzeros"] = np.percentile(df[df["conv"] != 0]["revenue"], percentile)

    return percentiles


def permutation_test(old: pd.DataFrame, new: pd.DataFrame, num_permutations=500):
    df = pd.DataFrame(columns=["statistic", "p_value"], index=old.columns)

    for col in old.columns:
        test = stats.permutation_test(
            (np.array(old[col], dtype=float), np.array(new[col], dtype=float)),
            statistic=lambda x, y: x.mean() - y.mean(),
            n_resamples=num_permutations,
        )
        df.loc[col] = test.statistic, test.pvalue
        print(f"{col} done")

    if ("conv" in old.columns) and ("revenue" in old.columns):
        test = stats.permutation_test(
            (
                np.array(old[old["conv"] != 0]["revenue"], dtype=float),
                np.array(new[new["conv"] != 0]["revenue"], dtype=float),
            ),
            statistic=lambda x, y: x.mean() - y.mean(),
            n_resamples=num_permutations,
        )
        df.loc["revenue_nonzeros"] = test.statistic, test.pvalue
        print(f"revenue_nonzeros done")

    return df


def t_test(old: pd.DataFrame, new: pd.DataFrame):
    df = pd.DataFrame(columns=["statistic", "p_value"], index=old.columns)

    for col in old.columns:
        df.loc[col] = stats.ttest_ind(
            np.array(old[col], dtype=float), np.array(new[col], dtype=float)
        )

    if ("conv" in old.columns) and ("revenue" in old.columns):
        df.loc["revenue_nonzeros"] = stats.ttest_ind(
            np.array(old[old["conv"] != 0]["revenue"], dtype=float),
            np.array(new[new["conv"] != 0]["revenue"], dtype=float),
        )

    return df


def reduce_uids(df: pd.DataFrame):
    return df.groupby("uid").agg(max)


if __name__ == "__main__":
    import utils

    import pandas as pd
    import numpy as np
    import seaborn as sns

    df = utils.generate_df(ids_num=int(1e5))
    old = df[df.group == 0].drop(columns=["uid", "group"])
    new = df[df.group == 1].drop(columns=["uid", "group"])

    print(permutation_test(old, new))
