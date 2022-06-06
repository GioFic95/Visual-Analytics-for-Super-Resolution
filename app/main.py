import itertools
import pprint
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import dash
from dash import dcc, Output, Input, State
from dash import html
import dash_bootstrap_components as dbc

from app.plots import parallel_plot, scatter_plot


def get_df(csv: Path, types: Dict[str, type]) -> pd.DataFrame:
    df = pd.read_csv(csv, dtype=types)
    df.query("mask == False", inplace=True)
    return df


def main(csv_avg: Path, csv_all: Path, highlights: List[str] = []):
    types = {"name": str, "ssim": float, "psnr_rgb": float, "psnr_y": float, "lpips": float,
             "type": str, "mask": bool, "category": str}
    metrics = ["ssim", "psnr_rgb", "psnr_y", "lpips"]

    app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY],
                    assets_folder="./resources", assets_url_path='/',
                    suppress_callback_exceptions=True)

    curr_dfp = get_df(csv_avg, types)
    curr_dfs = get_df(csv_all, types)
    queries = {"dataset": "", "compression": "", "parallel": ""}

    pp = parallel_plot(curr_dfp)
    metric_combos = [f"{m1} VS {m2}" for m1, m2 in itertools.combinations(metrics, 2)]
    last_m12 = [None, None]

    div_parallel = html.Div(dcc.Graph(config={'displayModeBar': False, 'doubleClick': 'reset'},
                                      figure=pp, id=f"my-graph-pp", style={'height': 500}),
                            className='row')
    div_scatter = html.Div([
        html.Div(id=f"my-div-sp", className='col-8'),
        html.Div(id=f"my-img", className='col-4')
    ], className='row')

    metrics_label = html.Label("Metrics:", style={'font-weight': 'bold', "text-align": "center"})
    metrics_dd = dcc.Dropdown(
                    id="metrics-dropdown",
                    options=metric_combos,
                    value="ssim VS psnr_rgb",
                    style={'width': '200px'}
    )
    metrics_div = html.Div([metrics_label, metrics_dd], className="col")

    dataset_label = html.Label("Training dataset:", style={'font-weight': 'bold'})
    dataset_radio = dcc.RadioItems({"isb": "F4K+", "saipem": "Saipem", "": "All"}, "", id="dataset-radio",
                                   className="form-check", labelStyle={'display': 'flex'},
                                   inputClassName="form-check-input", labelClassName="form-check-label")
    dataset_div = html.Div([dataset_label, dataset_radio], className="col")

    compression_label = html.Label("Compression type:", style={'font-weight': 'bold'})
    compression_radio = dcc.RadioItems({"img": "Image Compression", "vid": "Video Compression", "": "All"}, "",
                                       id="compression-radio", className="form-check", labelStyle={'display': 'flex'},
                                       inputClassName="form-check-input", labelClassName="form-check-label")
    compression_div = html.Div([compression_label, compression_radio], className="col")

    div_buttons = html.Div([dataset_div, compression_div, metrics_div], className="row", style={"margin": 15})

    div_title = html.Div(html.H1("Visual Analytics for Underwater Super Resolution"),
                         style={"margin-top": 30, "margin-left": 30})

    def make_query(avg: bool = False) -> str:
        if avg:
            query_list = [v for k, v in queries.items() if v != "" and k != "parallel"]
        else:
            query_list = [v for v in queries.values() if v != ""]
        query = " & ".join(query_list)
        print(query)
        return query

    @app.callback(
        Output('my-div-sp', 'children'),
        Output('my-graph-pp', 'figure'),
        Input('metrics-dropdown', 'value'),
        Input('dataset-radio', 'value'),
        Input('compression-radio', 'value')
    )
    def update_sp(drop_mc, radio_ds, radio_cp):
        print('update_sp', drop_mc, radio_ds, radio_cp)

        m1, m2 = str(drop_mc).split(" VS ")
        last_m12[0:2] = m1, m2
        queries["dataset"] = f"train == '{radio_ds}'" if radio_ds != "" else ""
        queries["compression"] = f"type == '{radio_cp}'" if radio_cp != "" else ""
        query = make_query()
        updated_dfs = curr_dfs.query(query) if len(query) > 0 else curr_dfs
        updated_dfp = curr_dfp.query(query) if len(query) > 0 else curr_dfp
        print(updated_dfs.shape, updated_dfp.shape)

        scatter = scatter_plot(updated_dfs, m1, m2, highlights)
        scatter.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        new_div = dcc.Graph(config={'displayModeBar': False, 'doubleClick': 'reset'},
                            figure=scatter, id=f"my-graph-sp")
        new_par = parallel_plot(updated_dfp)
        return new_div, new_par

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
                html.Div(f"{name} ({trace})", style={"margin-top": 10, "margin-bottom": 15}),
            ])
            return new_div
        else:
            return None

    @app.callback(
        Output('my-graph-sp', 'figure'),
        Input('my-graph-pp', 'restyleData'),
        State('my-graph-pp', 'figure'),
        State('my-graph-sp', 'figure')
    )
    def display_select_parallel(selection, pp_fig, sp_fig):
        print("selection:", selection)

        if selection is None:
            return sp_fig
        else:
            # par_coord_data = pp_fig['data'][0]
            # pprint.pprint(par_coord_data)
            curr_dims = pp_fig['data'][0].get('dimensions', [])
            dim = curr_dims[-1]
            assert dim['label'] == 'Name'
            traces = dim['ticktext']
            idxs = dim['tickvals']

            for dim in curr_dims:
                if dim['label'] == 'Name':
                    continue
                else:
                    try:
                        constraints = np.array(dim['constraintrange'])
                        vals = np.array(dim['values'])
                        if len(constraints.shape) == 1:
                            new_idxs = np.where((vals > constraints[0]) & (vals < constraints[1]))
                        elif len(constraints.shape) == 2:
                            new_idxs = np.array(0)
                            for c in constraints:
                                print(c)
                                new_idxs = np.union1d(np.where((vals > c[0]) & (vals < c[1])), new_idxs)
                        else:
                            raise ValueError
                        idxs = np.intersect1d(idxs, new_idxs)
                    except KeyError:
                        continue

            traces = [traces[i] for i in idxs]  # do NOT remove: it is used in the query!
            # print(traces, idxs)

            m1, m2 = last_m12
            queries["parallel"] = f"category in {[t for t in traces]}"
            updated_df = curr_dfs.query(make_query())
            return scatter_plot(updated_df, m1, m2, highlights)

    app.layout = html.Div([div_title, div_parallel, div_buttons, div_scatter])
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
