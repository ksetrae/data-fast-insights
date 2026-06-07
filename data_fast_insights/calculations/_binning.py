from typing import TYPE_CHECKING
import warnings
from contextlib import contextmanager

import optbinning.binning.metrics
import numpy as np
import pandas as pd
from optbinning import OptimalBinning



if TYPE_CHECKING:
    from data_fast_insights import BinaryDependenceModelData


# This is required due to compatibility issues between optbinning and newer versions of scikit-learn
@contextmanager
def safe_optbinning_patch():
    """
    Directly intercepts check_array inside optbinning's internal metrics module
    to map the removed 'force_all_finite' keyword to 'ensure_all_finite'.
    """
    original_check_array = optbinning.binning.metrics.check_array

    def patched_check_array(*args, **kwargs):
        if 'force_all_finite' in kwargs:
            kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
        return original_check_array(*args, **kwargs)

    optbinning.binning.metrics.check_array = patched_check_array
    try:
        yield
    finally:
        optbinning.binning.metrics.check_array = original_check_array


def get_optbinning_bins(df: pd.DataFrame, num_cols: list, y_col: str, min_bin_size: float = 0.05) -> dict:
    """
    Generates a dictionary matching scorecardpy sc.woebin structure
    including both 'bin' intervals and 'breaks' tracking metrics.
    """
    bins_dict = {}
    y_vector = df[y_col].values

    with safe_optbinning_patch():
        for col in num_cols:
            X_vector = df[col].values

            optb = OptimalBinning(
                name=col,
                dtype="numerical",
                solver="cp",
                min_bin_size=min_bin_size
            )
            optb.fit(X_vector, y_vector)

            splits = optb.splits

            if len(splits) == 0:
                valid_x = X_vector[~np.isnan(X_vector)]
                if len(valid_x) > 0:
                    splits = np.nanpercentile(valid_x, [25, 50, 75])
                    splits = np.unique(splits)
                else:
                    splits = np.array([])

            intervals = [-np.inf] + list(splits) + [np.inf]

            bin_labels = []
            break_labels = []

            for i in range(len(intervals) - 1):
                # 1. Create standard interval strings e.g. [-inf, 11.0)
                bin_labels.append(f"[{intervals[i]},{intervals[i + 1]})")

                # 2. Extract upper boundaries for your breaks tracking array
                upper_bound = intervals[i + 1]
                if upper_bound == np.inf:
                    break_labels.append('inf')
                elif upper_bound == -np.inf:
                    break_labels.append('-inf')
                else:
                    # Formats floats cleanly to strip extra floating-point precision decimals
                    break_labels.append(str(float(upper_bound)))

            # If missing rows exist, align 'missing' data to match structure
            if pd.Series(X_vector).isnull().any():
                bin_labels.append('missing')
                # Missing categories traditionally do not have evaluation splits/numerical bounds
                break_labels.append(np.nan)

            # Wrap variables into the Dataframe schema required by downstream functions
            bins_dict[col] = pd.DataFrame({
                'bin': bin_labels,
                'breaks': break_labels
            })

    return bins_dict


def make_bins(model_data: 'BinaryDependenceModelData', manual_breaks: dict = None) -> dict:
    """ Make bins for numeric variables of model_data, optimizing for Information Value
        based on created binary target

    Parameters
    ----------
    model_data

    Returns
    -------
    dict
        Info about bins, where keys are features, values are dataframes with data about bins
    """
    if not model_data.num_cols:
        warnings.warn('model_data.num_cols is not set')
        return dict()
    if model_data.is_data_converted:
        warnings.warn(
            "Features in model_data seem to be already converted to binary format, binning might be futile")

    if manual_breaks is not None:
        warnings.warn("Manual breaks are not supported currently, including it has no effect")

    num_cols = list(model_data.num_cols)

    dt = model_data.base_data[num_cols].join(model_data.data[model_data.y_binary_name])
    bins = get_optbinning_bins(
        df=dt,
        num_cols=num_cols,
        y_col=model_data.y_binary_name
    )

    return bins


def get_breaks(bins: dict) -> dict:
    breaks = {column: bins[column]['breaks'].tolist() for column in bins}
    return breaks
