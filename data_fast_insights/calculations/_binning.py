from typing import TYPE_CHECKING
import warnings
from contextlib import contextmanager

import optbinning.binning.metrics
import sklearn.utils.validation
import numpy as np
import pandas as pd
from optbinning import OptimalBinning



if TYPE_CHECKING:
    from data_fast_insights import BinaryDependenceModelData


BIN_DIGITS = 1


# This is required due to compatibility issues between optbinning and newer versions of scikit-learn
@contextmanager
def safe_optbinning_patch():
    original_check_array = sklearn.utils.validation.check_array

    def patched_check_array(*args, **kwargs):
        if 'force_all_finite' in kwargs:
            kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
        return original_check_array(*args, **kwargs)

    sklearn.utils.validation.check_array = patched_check_array

    import optbinning.binning.metrics
    import optbinning.binning.binning

    original_metrics_check = getattr(optbinning.binning.metrics, "check_array", None)
    original_binning_check = getattr(optbinning.binning.binning, "check_array", None)

    optbinning.binning.metrics.check_array = patched_check_array
    optbinning.binning.binning.check_array = patched_check_array

    try:
        yield
    finally:
        sklearn.utils.validation.check_array = original_check_array
        if original_metrics_check is not None:
            optbinning.binning.metrics.check_array = original_metrics_check
        if original_binning_check is not None:
            optbinning.binning.binning.check_array = original_binning_check


def _format_clean_float(val: float) -> str:
    """Helper to round to 2 decimals and strip trailing zeros (e.g., 5.0 -> '5')"""
    if val == np.inf:
        return 'inf'
    if val == -np.inf:
        return '-inf'

    rounded = round(float(val), 2)
    # If the float corresponds perfectly to an integer, drop the trailing .0
    if rounded.is_integer():
        return str(int(rounded))
    return str(rounded)


def get_optbinning_bins(
        df: pd.DataFrame,
        num_cols: list,
        y_col: str,
        max_pvalue: float | None = 0.05,
        min_bin_size: float = 0.05,
        max_n_bins: int | None = None,
        manual_breaks: dict = None
) -> dict:
    bins_dict = {}
    y_vector = df[y_col].values

    if manual_breaks is None:
        manual_breaks = {}

    with safe_optbinning_patch():
        for col in num_cols:
            X_vector = df[col].values

            kwargs = {
                "name": col,
                "dtype": "numerical",
                "solver": "cp",
                "min_bin_size": min_bin_size,
                "monotonic_trend": None,
                "min_event_rate_diff": 0.0,
                "gamma": 0.0,
                "max_n_bins": max_n_bins,
                "max_pvalue": max_pvalue,
                "max_pvalue_policy": "all"
            }

            if col in manual_breaks and manual_breaks[col] is not None:
                raw_splits = [x for x in manual_breaks[col] if x not in ['inf', '-inf', np.inf, -np.inf]]
                custom_splits = sorted(list(set(round(float(x), BIN_DIGITS) for x in raw_splits)))
                fixed_mask = [True] * len(custom_splits)

                kwargs["user_splits"] = custom_splits
                kwargs["user_splits_fixed"] = fixed_mask
                kwargs["monotonic_trend"] = None


            optb = OptimalBinning(**kwargs)

            optb.fit(X_vector, y_vector)

            raw_splits = optb.splits
            if len(raw_splits) == 0:
                valid_x = X_vector[~np.isnan(X_vector)]
                if len(valid_x) > 0:
                    raw_splits = np.nanpercentile(valid_x, )
                else:
                    raw_splits = np.array([])

            clean_splits = sorted(list(set(round(float(x), BIN_DIGITS) for x in raw_splits)))

            intervals = [-np.inf] + clean_splits + [np.inf]

            bin_labels = []
            break_labels = []

            for i in range(len(intervals) - 1):
                lower_str = _format_clean_float(intervals[i])
                upper_str = _format_clean_float(intervals[i + 1])

                bin_labels.append(f"[{lower_str}, {upper_str})")

                break_labels.append(upper_str)

            if pd.Series(X_vector).isnull().any():
                bin_labels.append('missing')
                break_labels.append(np.nan)

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
    num_cols = list(model_data.num_cols)

    dt = model_data.base_data[num_cols].join(model_data.data[model_data.y_binary_name])
    bins = get_optbinning_bins(
        df=dt,
        num_cols=num_cols,
        y_col=model_data.y_binary_name,
        manual_breaks=manual_breaks
    )

    return bins


def get_breaks(bins: dict) -> dict:
    breaks = {column: bins[column]['breaks'].tolist() for column in bins}
    return breaks
