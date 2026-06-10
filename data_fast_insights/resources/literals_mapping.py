SHOWCASE_LITERALS_MAPPING = {
    'count': lambda kwargs: 'Segment Size, Absolute',
    'worse_perc': lambda kwargs: 'Share of Objects Worse than the Total Mean, %',
    'better_perc': lambda kwargs: 'Share of Objects Better than the Total Mean (Segment Quality), %',
    'perc_of_total': lambda kwargs: 'Segment Size, %',
    'target_delta': lambda kwargs: f'{kwargs["target_name"]} Segment Difference from Total Mean, %',
    'target_delta_by_volume': lambda  kwargs: 'Group Effect'
}
