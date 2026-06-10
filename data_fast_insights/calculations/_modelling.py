from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.lines import segment_hits

from data_fast_insights import utils


if TYPE_CHECKING:
    from data_fast_insights import BinaryDependenceModelData


def calculate_dependence(
        model_data: 'BinaryDependenceModelData' = None,
        sort_from_best_to_worst: bool = True,

) -> pd.DataFrame:
    """ Calculate dependence on target for features in model_data

    Parameters
    ----------
    model_data
        If not set, return a dataframe with a row of default values
    sort_from_best_to_worst

    Returns
    -------
    pd.DataFrame
        DataFrame with data about dependence between features and target
        Columns description:
            count - sum of the binary feature values (size of the segment, absolute)
            worse_count - sum of the binary feature values where binary target equals 1
                (how much segment entries are lower than the selected
                target threshold (e.g. median or mean))
            worse_perc - (worse_count / count) * 100 (how bad the segment is, in percent).
                It represents the share of objects (rows) that are lower than the selected
                target threshold (e.g. median or mean) of all segment objects.
            better_perc - (100 - worse_perc).
            perc_of_total - segment share of total data, in percent (size of the segment, relative).
                Total "perc_of_total" of all segments across one base (original) feature equals 1.
            target_delta - how much target mean of this segment differs from total target mean, in percent
            target_delta_by_volume - (target_delta * perc_of_total) / 100
                Used to estimate total effect of the segment with consideration how large it is.
                Small segments can differ significantly, but be insignificant itself
            base_col - parent feature for segment.
                If the binary feature is a combination of multiple binary features,
                it contains json array of parent binary features
            base_breaks - chosen breaks of intervals of the parent feature (if parent feature is numeric)
            base_range - min and max values of the parent feature (if parent feature is numeric)
            base_cats - all possible categories of the parent feature (if parent feature is categorical)

    """
    if model_data is None:
        return pd.DataFrame.from_dict(
            {'count': 0, 'worse_count': 0, 'worse_perc': 0, 'better_perc': 0, 'perc_of_total': 0,
             'target_delta': 0, 'target_delta_by_volume': 0,
             'base_col': '', 'base_breaks': list(), 'base_range': list(), 'base_cats': list(),
             'target_mean': 0, 'target_median': 0},
            orient='index')

    df_features = model_data.data.drop(model_data.y_name, axis=1)
    res_total = pd.DataFrame(
        df_features.sum(axis=0), columns=['count']).sort_values(by='count', ascending=False)

    df_low = df_features[df_features[model_data.y_binary_name] == 1].drop(model_data.y_binary_name, axis=1).copy()
    res_low = pd.DataFrame(df_low.sum(axis=0), columns=['worse_count'])
    res_low = pd.merge(res_total, res_low, left_index=True, right_index=True)
    res_low['worse_perc'] = (res_low['worse_count'] / res_low['count']) * 100
    res_low['better_perc'] = 100 - res_low['worse_perc']
    res_low['perc_of_total'] = (res_low['count'] / model_data.data.shape[0]) * 100
    res_low['target_delta'] = np.nan
    res_low['target_delta_by_volume'] = np.nan

    res_low['base_col'] = ''
    res_low['base_breaks'] = ''
    res_low['base_breaks'] = res_low['base_breaks'].astype(object)
    res_low['base_range'] = ''
    # res_low['base_central_value'] = np.nan
    # res_low['base_central_value'] = res_low['base_central_value'].astype(object)
    # res_low['base_min'] = np.nan
    # res_low['base_max'] = np.nan
    res_low['base_cats'] = ''
    res_low['base_cats'] = res_low['base_cats'].astype(object)
    res_low['target_mean'] = np.nan
    res_low['target_median'] = np.nan

    # TODO: change from .iterrows() to faster type of iterations (e.g. zip() on series?)
    total_mean = model_data.data[model_data.y_name].mean()
    for i, row in res_low.iterrows():
        segment_target = model_data.data[model_data.data[i] == 1][model_data.y_name]
        segment_target_delta = ((segment_target.mean() / total_mean) - 1) * 100.0
        res_low.at[i, 'target_delta'] = segment_target_delta
        res_low.at[i, 'target_mean'] = segment_target.mean()
        res_low.at[i, 'target_median'] = segment_target.median()
        res_low.at[i, 'target_delta_by_volume'] = (segment_target_delta * res_low.at[i, 'perc_of_total']) / 100.0

        if i in model_data.col_links:
            base_col = model_data.col_links[i]
            res_low.at[i, 'base_col'] = base_col
            if base_col in model_data.num_cols:
                res_low.at[i, 'base_range'] = str([model_data.base_data[base_col].min(),
                                                   model_data.base_data[base_col].max()])
                res_low.at[i, 'base_breaks'] = model_data.bins[base_col]['breaks'].tolist()
            elif base_col in model_data.cat_cols:
                res_low.at[i, 'base_cats'] = model_data.base_data[base_col].unique()
        else:
            raise ValueError("Segment name not found in links")

    sort_ascending = model_data.inverse_goal
    if not sort_from_best_to_worst:
        sort_ascending = not sort_ascending

    res_low = res_low.sort_values(by='target_delta_by_volume', ascending=sort_ascending)

    return res_low

def compare_intervals(selected: str, model_data: 'BinaryDependenceModelData') -> pd.DataFrame:
    """ Compare how changing certain values to other interval of same feature would affect the target.

        Supposing model_data.data and model_data.base_data have same points on same indices
        (which unless these attributes are modified manually is true)

    Parameters
    ----------
    selected
        Segment (binary feature name) that needs to be compared to other segments
    model_data

    Returns
    -------
    pd.DataFrame
        Results of comparison
        Columns description:
        (note: <metric_name> is a metric describing the column, e.g. 'mode' for categorical, 'mean' for numeric)
            old_col - segment being compared (current segment)
            old_<metric_name> - metric of the current segment (e.g. for 'trial_period_[-inf, 1)
            old_base_<metric_name> - metric of the base feature of the current segment (e.g. for 'trial_period')
            new_col - segment that old_col is being compared to (new segment)
            new_<metric_name> - metric of the new segment
            new_base_<metric_name> - metric of the parent feature of the new segment
            total_target_change_perc - how much this substitution changes total target value (on all data), in percent
    """
    # TODO: make it so model_data.data and model_data.base_data don't have to have same points on same indices
    #  or make it explicit.

    if selected not in model_data.data.columns:
        raise ValueError(f"'{selected}' feature not found in model_data.data;"
                         + " make sure you pass a binary segment name, not the original feature name")

    sel_interval_indices = model_data.data.loc[model_data.data[selected] == 1].index
    base_col = model_data.col_links[selected]
    comparison = [binary for binary, base in model_data.col_links.items() if base == base_col and binary != selected]

    # pd_metrics_attr must be a name of a pd.DataFrame method calculating some metric.
    pd_metrics_attr = utils.choose_central_tendency_metric(base_col, model_data)
    comp_df = pd.DataFrame(columns=['old_col',
                                    'old_' + pd_metrics_attr,
                                    'old_base_' + pd_metrics_attr,
                                    'new_col',
                                    'new_' + pd_metrics_attr,
                                    'new_base_' + pd_metrics_attr,
                                    'total_target_change_perc'])
    current_comp_data = {'old_col': selected,
                         'old_' + pd_metrics_attr: model_data.base_data.loc[sel_interval_indices, :][
                             base_col].__getattribute__(pd_metrics_attr)(),
                         'old_base_' + pd_metrics_attr:
                             model_data.base_data[base_col].__getattribute__(pd_metrics_attr)()}

    for index, compare_to in enumerate(comparison):
        comp_interval_indices = model_data.data.loc[model_data.data[compare_to] == 1].index

        df_int_tmp = model_data.data.copy()
        df_int_tmp.loc[sel_interval_indices,
                       [model_data.y_name]] = df_int_tmp.loc[comp_interval_indices, :][model_data.y_name].mean()

        df_base_tmp = model_data.base_data.copy()

        df_base_tmp.loc[sel_interval_indices, [base_col]] = df_base_tmp.loc[comp_interval_indices, :][
            base_col].__getattribute__(pd_metrics_attr)()
        total_target_increase = (df_int_tmp[model_data.y_name].sum() / model_data.data[model_data.y_name].sum() - 1)

        current_comp_data['new_col'] = compare_to
        current_comp_data['new_' + pd_metrics_attr] = model_data.base_data.loc[comp_interval_indices, :][
            base_col].__getattribute__(pd_metrics_attr)()
        current_comp_data['new_base_' + pd_metrics_attr] = df_base_tmp[base_col].__getattribute__(pd_metrics_attr)()

        current_comp_data['total_target_change_perc'] = total_target_increase * 100
        comp_df = comp_df.append(pd.DataFrame(current_comp_data, index=[index]))
    return comp_df
