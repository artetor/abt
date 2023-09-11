import numpy as np
import pandas as pd
from hashlib import sha256


def generate_uids(ids_num: int = int(1e5), mean: float = 1, sigma: float = 2):
    """
    Generate user ids (uids) array. Frequency of each uid is sampled from lognormal distribution.
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


def generate_convs(
    sample_size: int = int(2e7), old_proba: float = 7.6e-4, new_proba: float = 7.9e-4
):
    """
    generate array of conversions.
    ## Params
    * sample_size: size of each array
    * old_proba: CVR before treatment
    * new_proba: CVR after treatment
    """
    old = np.random.choice([1, 0], size=sample_size, p=[old_proba, 1 - old_proba])
    new = np.random.choice([1, 0], size=sample_size, p=[new_proba, 1 - new_proba])

    return old, new


def generate_df(
    ids_num: int,
    mean: float = 1,
    sigma: float = 2,
    old_proba: float = 7.6e-4,
    new_proba: float = 7.9e-4,
):
    """
    generate uids, old and new conversions as dataframe.
    """
    uids_arr, _ = generate_uids(ids_num, mean, sigma)

    old_convs, new_convs = generate_convs(len(uids_arr), old_proba, new_proba)

    # print(len(uids_arr))
    return pd.DataFrame().from_dict(
        {"uid": uids_arr, "old_convs": old_convs, "new_convs": new_convs}
    )
