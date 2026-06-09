import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def create_feature_segments_plot_plotly(info_df: pd.DataFrame) -> go.Figure:
    df = info_df.reset_index()
    index_col_name = df.columns[0]

    unique_features = sorted(df['base_col'].unique().tolist())
    if not unique_features:
        raise ValueError("The provided info_df does not contain any valid 'base_col' entries.")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Target Mean vs Segment Volume Share", "Share of Better Objects vs Segment Volume Share"),
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
    )

    buttons = []
    traces_per_feature = 4
    total_features = len(unique_features)

    for f_idx, feature in enumerate(unique_features):
        feature_df = df[df['base_col'] == feature].copy()

        # 2. Extract and preserve the raw interval strings for the x-axis exactly as they are ordered
        clean_xticks = []
        for raw_name in feature_df[index_col_name]:
            prefix = f"{feature}_"
            clean_xticks.append(str(raw_name)[len(prefix):] if str(raw_name).startswith(prefix) else str(raw_name))

        volume_pct = feature_df['perc_of_total'] if 'perc_of_total' in feature_df else [0] * len(clean_xticks)
        abs_counts = feature_df['count'] if 'count' in feature_df else [0] * len(clean_xticks)

        fig.add_trace(
            go.Bar(
                x=clean_xticks, y=volume_pct,
                name="Volume Share",
                marker=dict(color='rgba(180, 180, 180, 0.25)'),
                customdata=abs_counts,
                hovertemplate="<b>Volume Share</b>: %{y:.1f}%<br><b>Absolute Count</b>: %{customdata:,}<extra></extra>",
                visible=False, showlegend=(f_idx == 0)
            ),
            row=1, col=1, secondary_y=True
        )

        fig.add_trace(
            go.Scatter(
                x=clean_xticks, y=feature_df['target_mean'],
                mode='lines+markers', name="Target Mean",
                line=dict(width=3, color='#2b5c8f', dash='dot'),
                marker=dict(size=12, symbol='circle'),
                hovertemplate="<b>Target Mean</b>: %{y:.4f}<extra></extra>",
                visible=False, showlegend=(f_idx == 0)
            ),
            row=1, col=1, secondary_y=False
        )

        fig.add_trace(
            go.Bar(
                x=clean_xticks, y=volume_pct,
                name="Volume Share",
                marker=dict(color='rgba(180, 180, 180, 0.25)'),
                customdata=abs_counts,
                hovertemplate="<b>Volume Share</b>: %{y:.1f}%<br><b>Absolute Count</b>: %{customdata:,}<extra></extra>",
                visible=False, showlegend=False
            ),
            row=2, col=1, secondary_y=True
        )

        high_perc_val = feature_df['high_perc'] if 'high_perc' in feature_df else [0] * len(clean_xticks)
        fig.add_trace(
            go.Scatter(
                x=clean_xticks, y=high_perc_val,
                mode='lines+markers', name="% 'Better' Objects",
                line=dict(width=3, color='#e67e22', dash='solid'),
                marker=dict(size=12, symbol='diamond'),
                hovertemplate="<b>% Better than Pivot</b>: %{y:.1f}%<extra></extra>",
                visible=False, showlegend=(f_idx == 0)
            ),
            row=2, col=1, secondary_y=False
        )

        visibility_mask = [False] * (total_features * traces_per_feature)
        start_trace = f_idx * traces_per_feature
        visibility_mask[start_trace: start_trace + traces_per_feature] = [True, True, True, True]

        buttons.append(dict(
            label=feature,
            method="update",
            args=[{"visible": visibility_mask}]
        ))

    if len(fig.data) >= traces_per_feature:
        for i in range(traces_per_feature):
            fig.data[i].visible = True

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=750,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),

        yaxis=dict(
            title=dict(text="Target Mean Metric", font=dict(color="#2b5c8f")),
            tickfont=dict(color="#2b5c8f")
        ),
        yaxis2=dict(
            title=dict(text="Volume Share (%)", font=dict(color="gray")),
            tickfont=dict(color="gray"), range=[0, 100], showgrid=False
        ),

        yaxis3=dict(
            title=dict(text="% Better than Pivot", font=dict(color="#e67e22")),
            tickfont=dict(color="#e67e22"), range=[0, 100]
        ),
        yaxis4=dict(
            title=dict(text="Volume Share (%)", font=dict(color="gray")),
            tickfont=dict(color="gray"), range=[0, 100], showgrid=False
        ),

        updatemenus=[dict(
            active=0, buttons=buttons, direction="down",
            pad={"r": 10, "t": 10}, showactive=True,
            x=0.0, xanchor="left", y=1.15, yanchor="top"
        )]
    )

    fig.update_xaxes(
        type='category',
        categoryorder='array',
        tickangle=-30,
        row=2, col=1
    )

    return fig
