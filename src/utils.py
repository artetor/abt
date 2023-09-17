from typing import Union
import numpy as np
import pandas as pd
from hashlib import sha256


def generate_uids(ids_num: int = int(1e5), mean: float = 1, sigma: float = 2):
    """
    Generate user ids (uids) array. Frequency of each uid is sampled from *almost* lognormal distribution.
    ## Params
    * ids_num: number of uids
    * mean, sigma: params of the lognormal distribution from which frequency of each uid is sampled
    """
    uids_count_arr = np.random.lognormal(mean=mean, sigma=sigma, size=ids_num).astype(int)
    uids_count_arr = np.where(
        (uids_count_arr > 1000) | (uids_count_arr == 0),
        np.random.randint(1, 30, size=uids_count_arr.shape),
        uids_count_arr,
    )
    uids_arr = np.array(
        [
            sha256(str(i).encode()).hexdigest()
            for i, val in enumerate(uids_count_arr)
            for _ in range(val)
        ]
    )

    return uids_arr, uids_count_arr


def generate_convs(sample_size: int = int(2e7), proba: float = 7.9e-4):
    """
    generate array of conversions.
    ## Params
    * sample_size: size of each array
    * proba: CVR before treatment
    """
    convs = np.random.choice([1, 0], size=sample_size, p=[proba, 1 - proba])

    return convs


def generate_revenue(conv_vector: Union[np.ndarray, pd.Series], mean: float = 6, sigma: float = 1):
    """
    Generate revenue sampled from lognormal distribution.
    ## Params
    * conv_vector: boolean vector with converions info
    * mean, sigma: params of the lognormal distribution from which revenue is sampled
    """
    lognormal_samples = np.random.lognormal(mean, sigma, conv_vector.shape)
    revenues = np.where(conv_vector == 1, lognormal_samples, conv_vector)

    return revenues


def generate_df(
    ids_num: int,
    uid_mean: float = 1,
    uid_sigma: float = 2,
    old_conv_proba: float = 7.6e-4,
    new_conv_proba: float = 7.9e-4,
    old_rev_mean: float = 6,
    old_rev_sigma: float = 1,
    new_rev_mean: float = 6,
    new_rev_sigma: float = 1.2,
):
    """
    generate uids, split them to 2 groups, genereate conv and revenue for each group.
    """
    df = pd.DataFrame()
    df["uid"], _ = generate_uids(ids_num, uid_mean, uid_sigma)
    df["group"] = df["uid"].apply(lambda x: hash(x) % 2)
    df["conv"] = None

    def generate_conv_revenue(df, group, conv_proba, rev_mean, rev_sigma):
        df.loc[df.group == group, "conv"] = generate_convs(
            len(df[df.group == group]), proba=conv_proba
        )
        df.loc[df.group == group, "revenue"] = generate_revenue(
            df.loc[df.group == group, "conv"], mean=rev_mean, sigma=rev_sigma
        )

    generate_conv_revenue(
        df=df,
        group=0,
        conv_proba=old_conv_proba,
        rev_mean=old_rev_mean,
        rev_sigma=old_rev_sigma,
    )
    generate_conv_revenue(
        df=df,
        group=1,
        conv_proba=new_conv_proba,
        rev_mean=new_rev_mean,
        rev_sigma=new_rev_sigma,
    )

    return df
