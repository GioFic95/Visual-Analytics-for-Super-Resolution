from typing import List
import pandas as pd
import plotly.graph_objects as go


def parallel_plot(df: pd.DataFrame, title: str):
    if len(df) == 0:
        raise ValueError("\n***", title, "***   --> empty dataframe")

    dfc = df.copy()
    dfc["sum"] = dfc.sum(numeric_only=True)
    dfc.sort_values(by="sum", inplace=True)
    names_ids = list(range(len(dfc["name"])))

    return go.Figure(
        data=go.Parcoords(
            dimensions=list([
                dict(range=[dfc['lpips'].max(), dfc['lpips'].min()],
                     label='LPIPS', values=dfc['lpips']),
                dict(range=[dfc['ssim'].min(), dfc['ssim'].max()],
                     label='SSIM', values=dfc['ssim']),
                dict(range=[dfc['psnr_y'].min(), dfc['psnr_y'].max()],
                     label='PSNR Y', values=dfc['psnr_y']),
                dict(range=[dfc['psnr_rgb'].min(), dfc['psnr_rgb'].max()],
                     label='PSNR RGB', values=dfc['psnr_rgb']),
                dict(range=[0, len(dfc) - 1],
                     tickvals=names_ids, ticktext=dfc['name'],
                     label='Name', values=names_ids)
            ]),
            line=dict(color=names_ids, autocolorscale=True)
        ),
        layout=dict(title=title)
    )


def scatter_plot(df: pd.DataFrame, title: str, x: str, y: str, highlights: List[str] = []):
    scatter = go.Figure()

    for category, dfg in df.groupby("category"):
        sizes = [10 if n in highlights else 5 for n in dfg["name"]]
        widths = [1 if n in highlights else 0 for n in dfg["name"]]
        opacs = [.9 if n in highlights else .6 for n in dfg["name"]]
        symbols = ["star" if n in highlights else
                   "circle" for n in dfg["name"]]

        scatter.add_trace(go.Scatter(
            x=dfg[x],
            y=dfg[y],
            mode='markers',
            text=dfg["name"],
            name=category,
            marker=dict(opacity=opacs,
                        size=sizes,
                        symbol=symbols,
                        line=dict(width=widths, color="black"),
                        )))

    scatter.update_layout(
        title=title,
        xaxis_title=x,
        yaxis_title=y,
        clickmode='event+select',
        height=800,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="left",
            x=0,
            font=dict(size=10)
        )
    )

    return scatter
