import plotly.graph_objects as go
import pandas as pd


def create_feature_segments_plot_plotly(info_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    df = info_df.reset_index()
    index_col_name = df.columns[0]

    unique_features = df['base_col'].unique().tolist()

    if not unique_features:
        raise ValueError("The provided info_df does not contain any valid 'base_col' entries.")

    buttons = []
    trace_counter = 0

    for feature in unique_features:
        feature_df = df[df['base_col'] == feature].copy()


        clean_xticks = []
        for raw_name in feature_df[index_col_name]:
            prefix = f"{feature}_"
            if str(raw_name).startswith(prefix):
                interval_str = str(raw_name)[len(prefix):]
            else:
                interval_str = str(raw_name)
            clean_xticks.append(interval_str)

        fig.add_trace(
            go.Scatter(
                x=clean_xticks,
                y=feature_df['target_mean'],
                mode='lines+markers',
                name=feature,
                visible=False,

                line=dict(width=3, color='#2b5c8f', dash='dot'),

                marker=dict(size=12, symbol='circle', color='#2b5c8f')
            )
        )

        visibility_mask = [False] * len(unique_features)
        visibility_mask[trace_counter] = True

        buttons.append(dict(
            label=feature,
            method="update",
            args=[
                {"visible": visibility_mask},
                {"title": f"Target Mean Across Segments: {feature}"}
            ]
        ))

        trace_counter += 1

    if fig.data:
        fig.data[0].visible = True

    fig.update_layout(
        title=f"Target Mean Across Segments: {unique_features[0]}",
        xaxis_title="Feature Segment Intervals",
        yaxis_title="Target Variable Mean",
        template="plotly_white",
        hovermode="x unified",

        updatemenus=[dict(
            active=0,
            buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.0,
            xanchor="left",
            y=1.15,
            yanchor="top"
        )]
    )

    fig.update_xaxes(tickangle=-30)

    return fig
