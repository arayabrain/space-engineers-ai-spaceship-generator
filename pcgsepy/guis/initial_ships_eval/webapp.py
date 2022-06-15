import json
from datetime import datetime
from multiprocessing import Event
from typing import Dict, List, Tuple
import os, pathlib

import dash
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from pcgsepy.config import BIN_POP_SIZE, CS_MAX_AGE, N_GENS_ALLOWED
from pcgsepy.common.jsonifier import json_dumps, json_loads
from pcgsepy.lsystem.rules import StochasticRules
from pcgsepy.lsystem.solution import CandidateSolution
from pcgsepy.mapelites.emitters import (ContextualBanditEmitter,
                                        HumanPrefMatrixEmitter, RandomEmitter)
from pcgsepy.mapelites.map import MAPElites, get_structure


block_to_colour = {
    # colours from https://developer.mozilla.org/en-US/docs/Web/CSS/color_value
    'LargeBlockArmorCorner': '#778899',
    'LargeBlockArmorSlope': '#778899',
    'LargeBlockArmorCornerInv': '#778899',
    'LargeBlockArmorBlock': '#778899',
    'LargeBlockGyro': '#2f4f4f',
    'LargeBlockSmallGenerator': '#ffa07a',
    'LargeBlockSmallContainer': '#008b8b',
    'OpenCockpitLarge': '#32cd32',
    'LargeBlockSmallThrust': '#ff8c00',
    'SmallLight': '#fffaf0',
    'Window1x1Slope': '#fffff0',
    'Window1x1Flat': '#fffff0',
    'LargeBlockLight_1corner': '#fffaf0'
}


app = dash.Dash(__name__,
                title='Spasceships evaluator',
                external_stylesheets=[
                    'https://codepen.io/chriddyp/pen/bWLwgP.css'],
                update_title=None)


def set_app_layout(spaceships: List[str]):
    description_str, help_str = '', ''
    
    curr_dir = pathlib.Path(__file__).parent.resolve()
    
    with open(curr_dir.joinpath('assets/description.md'), 'r') as f:
        description_str = f.read()
    with open(curr_dir.joinpath('assets/help.md'), 'r') as f:
        help_str = f.read()
    
    app.layout = html.Div(children=[
        # HEADER
        html.Div(children=[
            html.H1(children='🚀Space Engineers🚀 Spaceships evaluator',
                    className='title'),
            dcc.Markdown(children=description_str,
                         className='page-description'),
        ],
            className='header'),
        html.Br(),
        # BODY
        html.Div(children=[
            # current spaceship counter display
            html.Div(children=[
                html.H1(children=f'1 / {len(spaceships)}',
                        id='spaceship-current')
                ]),
            html.Br(),
            # current spaceship content display + properties
            html.Div(children=[
                html.H6(children=spaceships[0],
                        id='spaceship-content')
                ]),
            html.Br(),
            # slider
            html.Div(children=[
                dcc.Slider(-2, 2, 0.5,
                           value=0,
                           id='value-slider'
                           ),
                ]),
            html.Br(),
            # prev/next and save buttons
            html.Div(children=[
                html.Button('<',
                            id='prev-btn'),
                html.Button('>',
                            id='next-btn'
                            ),
                html.Div(children=[
                    html.Button('SAVE',
                            id='save-btn',
                            disabled=True),
                    dcc.Download(id='download-values')
                ])
                ]),
            ]),
        html.Br(),
        # FOOTER
        html.Div(children=[
            html.H6(children='Help',
                    className='section-title'),
            dcc.Markdown(help_str,
                         className='page-description')
        ],
            className='footer'),
        dcc.Store(id='spaceships',
                  data=json.dumps(spaceships)),
        dcc.Store(id='current-idx',
                  data=0),
        dcc.Store(id='values',
                  data=','.join(['0' for _ in range(len(spaceships))]))
    ])


@app.callback(
    Output("download-values", "data"),
    Input("save-btn", "n_clicks"),
    State('values', 'data'),
    State('spaceships', 'data'),
    prevent_initial_call=True,
)
def download_values(n_clicks,
                    values,
                    spaceships):
    spaceships = json.loads(spaceships)
    values = [float(x) for x in values.split(',')]
    
    t = datetime.now().strftime("%Y%m%d%H%M%S")
    fname = f'{t}'
    content = {k:v for k, v in zip(spaceships, values)}
    return dict(content=content, filename=f'{fname}.log')


@app.callback(Output('current-idx', 'data'),
              Output('values', 'data'),
              Output('spaceship-current', 'children'),
              Output('spaceship-content', 'children'),
              Output('value-slider', 'value'),
              Input('value-slider', 'value'),
              Input('prev-btn', 'n_clicks'),
              Input('next-btn', 'n_clicks'),
              Input('save-btn', 'n_clicks'),
              State('spaceships', 'data'),
              State('current-idx', 'data'),
              State('values', 'data'))
def general_callback(slider_value, prev_n_clicks, next_n_clicks, save_n_clicks,
                     spaceships, current_idx, values):
    spaceships = json.loads(spaceships)
    current_idx = int(current_idx)
    values = [float(x) for x in values.split(',')]
    
    ctx = dash.callback_context

    if not ctx.triggered:
        event_trig = None
    else:
        event_trig = ctx.triggered[0]['prop_id'].split('.')[0]

    if event_trig == 'prev-btn':
        current_idx -= 1
        current_idx = max(0, current_idx)
    elif event_trig == 'next-btn':
        current_idx += 1
        current_idx = min(len(values) - 1, current_idx)
    elif event_trig is not None and 'value-slider' in event_trig:
        values[current_idx] = slider_value
    
    return str(current_idx), ','.join([str(x) for x in values]), f'{current_idx + 1} / {len(values)}', spaceships[current_idx], values[current_idx]