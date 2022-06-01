import itertools
from pathlib import Path
from typing import List, Dict

import pandas as pd
import dash
from dash import dcc, Output, Input
from dash import html
import dash_bootstrap_components as dbc

from app.plots import parallel_plot, scatter_plot


def get_dfs(csv: Path, titles: List[str], types: Dict[str, type]):
    df = pd.read_csv(csv, dtype=types)

    df_img = df.loc[(df['type'] == 'img') & (df['mask'] == False)]
    df_img_mask = df.loc[(df['type'] == 'img') & (df['mask'] == True)]
    df_vid = df.loc[(df['type'] == 'vid') & (df['mask'] == False)]
    df_vid_mask = df.loc[(df['type'] == 'vid') & (df['mask'] == True)]
    df.loc[df_img.index, "type_mask"] = titles[0]
    df.loc[df_img_mask.index, "type_mask"] = titles[1]
    df.loc[df_vid.index, "type_mask"] = titles[2]
    df.loc[df_vid_mask.index, "type_mask"] = titles[3]
    return [df_img, df_img, df_img_mask, df_vid, df_vid_mask]


def main(csv_avg: Path, csv_all: Path, highlights: List[str] = []):
    titles = ["Tests on images", "Tests on images with masks", "Tests on videos", "Tests on videos with masks"]
    types = {"name": str, "ssim": float, "psnr_rgb": float, "psnr_y": float, "lpips": float,
             "type": str, "mask": bool, "category": str}
    metrics = ["ssim", "psnr_rgb", "psnr_y", "lpips"]

    app = dash.Dash(external_stylesheets=["bWLwgP.css", dbc.themes.FLATLY],
                    assets_folder="./resources", assets_url_path='/',
                    suppress_callback_exceptions=True)

    dfs1 = get_dfs(csv_avg, titles, types)
    dfs2 = get_dfs(csv_all, titles, types)

    pp = parallel_plot(dfs1[0], "images")  # todo extend to other dfs
    scatters = dict()
    for m1, m2 in itertools.combinations(metrics, 2):
        metric_combo = f"{m1} VS {m2}"
        title = f"images ({metric_combo})"
        scatters[metric_combo] = scatter_plot(dfs2[0], title, m1, m2, highlights)  # todo extend to other dfs

    div_parallel = html.Div(dcc.Graph(config={'displayModeBar': False, 'doubleClick': 'reset'},
                                      figure=pp, id=f"my-graph-pp"),
                            className='row')
    div_scatter = html.Div([
        html.Div(id=f"my-div-sp", className='eight columns'),
        html.Div(id=f"my-img", className='four columns')
    ], className='row')

    dropdown = dcc.Dropdown(
                    id="metrics-dropdown",
                    options=[
                        {'label': mc, 'value': mc} for mc in scatters.keys()
                    ],
                    value="ssim VS psnr_rgb",
                    style={'width': '200px'}
    )

    @app.callback(
        Output('my-div-sp', 'children'),
        [Input('metrics-dropdown', 'value')]
    )
    def update_fig(drop_mc):
        new_plot = dcc.Graph(config={'displayModeBar': False, 'doubleClick': 'reset'},
                             figure=scatters[drop_mc], id=f"my-graph-sp"),
        return new_plot

    @app.callback(
        Output('my-img', 'children'),
        Input('my-graph-sp', 'clickData'),
        Input('my-graph-sp', 'figure'),
    )
    def display_click_data(click_data, graph):
        if click_data is not None:
            trace = graph['data'][click_data['points'][0]['curveNumber']]['name']
            name = click_data['points'][0]['text']
            gt_name = name.split("_")[0] + ".png"
            print("click:", click_data, "\n", trace, "\n")
            new_div = html.Div([
                html.Img(src=f"imgs/{gt_name}", height=395),
                html.Img(src=f"imgs/{name}", height=395),
                html.Br(),
                html.Span(f"{name} ({trace})"),
            ])
            return new_div
        else:
            return None

    @app.callback(
        Output('my-graph-sp', 'figure'),
        Input('my-graph-pp', 'restyleData'),
        Input('my-graph-sp', 'figure')
    )
    def callback(selection, g):
        print("selection:", selection)
        return g

    app.layout = html.Div([div_parallel, dropdown, div_scatter])
    app.run_server(debug=True, use_reloader=False)


if __name__ == '__main__':
    highlights = [
        "00000_BSRGAN_isb_7_3.png",
        "00000_BSRGAN_isb_7_freeze_99.png",
        "00000_BSRGAN_isb_7_t5_78.png",
        "00010_BSRGAN_isb_7_t5_78.png",
        "00011_BSRGAN_isb_2_3.png",
        "00029_BSRGAN_isb_7_freeze_12.png",
        "00029_BSRGAN_isb_7_freeze_99.png",
        "00029_BSRGAN_isb_7_t5_30.png",
        "00069_BSRGAN_isb_2_3.png",
        "00069_BSRGAN_isb_7_freeze_99.png",
        "00438_BSRGAN_isb_7_t5_78.png"
    ]
    main(Path("./resources/test_results_isb.csv"), Path("./resources/test_results_all_isb.csv"), highlights)
