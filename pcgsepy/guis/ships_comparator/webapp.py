import os
import sys


def resource_path(relative_path):
# get absolute path to resource
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


import base64
import json
import random
from typing import Dict, List

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, dcc, html
from dash.dependencies import Input, Output, State
from pcgsepy.common.api_call import block_definitions
from pcgsepy.evo.genops import expander
from pcgsepy.guis.ships_comparator.modals_msgs import (rankings_assigned,
                                                       scores_different_error)
from pcgsepy.hullbuilder import HullBuilder
from pcgsepy.lsystem.solution import CandidateSolution
from pcgsepy.mapelites.emitters import *
from pcgsepy.mapelites.emitters import (ContextualBanditEmitter,
                                        HumanPrefMatrixEmitter, RandomEmitter)
from pcgsepy.setup_utils import get_default_lsystem

used_ll_blocks = [
    'MyObjectBuilder_CubeBlock_LargeBlockArmorCornerInv',
    'MyObjectBuilder_CubeBlock_LargeBlockArmorCorner',
    'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope',
    'MyObjectBuilder_CubeBlock_LargeBlockArmorBlock',
    'MyObjectBuilder_Gyro_LargeBlockGyro',
    'MyObjectBuilder_Reactor_LargeBlockSmallGenerator',
    'MyObjectBuilder_CargoContainer_LargeBlockSmallContainer',
    'MyObjectBuilder_Cockpit_OpenCockpitLarge',
    'MyObjectBuilder_Thrust_LargeBlockSmallThrust',
    'MyObjectBuilder_InteriorLight_SmallLight',
    'MyObjectBuilder_CubeBlock_Window1x1Slope',
    'MyObjectBuilder_CubeBlock_Window1x1Flat',
    'MyObjectBuilder_InteriorLight_LargeBlockLight_1corner'
]

lsystem = get_default_lsystem(used_ll_blocks=used_ll_blocks)

expander.initialize(rules=lsystem.hl_solver.parser.rules)

hull_builder = HullBuilder(erosion_type='bin',
                           apply_erosion=True,
                           apply_smoothing=True)

emitters = ['Human', 'Random', 'Greedy', 'Contextual Bandit']
progress = -1

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


def _get_colour_mapping(block_types: List[str]) -> Dict[str, str]:
    colour_map = {}
    for block_type in block_types:
        c = block_to_colour.get(block_type, '#ff0000')
        if block_type not in colour_map.keys():
            colour_map[block_type] = c
    return colour_map


app = dash.Dash(__name__,
                title='Spaceships Ranker',
                external_stylesheets=[dbc.themes.DARKLY],
                assets_folder=resource_path("assets"),
                update_title=None)


def get_content_div(content_n: int,
                    tot_content: int) -> html.Div:
    w = 100 // tot_content
    return html.Div(children=[
        # title
        html.Div(children=[
            html.H1(children=f'Spaceship {chr(ord("A") + content_n)}',
                    style={'text-align': 'center'})
            ]),
        html.Br(),
        # spaceship content display + properties
        # CONTENT PLOT
        html.Div(children=[
            dcc.Graph(id={'type': "spaceship-content", 'index': content_n},
                      figure=px.scatter_3d(x=np.zeros(0, dtype=object),
                                           y=np.zeros(0, dtype=object),
                                           z=np.zeros(0, dtype=object),
                                           title='',
                                           template='plotly_dark'),
                      config={
                          'displayModeBar': False,
                          'displaylogo': False}),
            ],
                className='content-div',
                style={'width': '100%'}),
        ],
                        style={'width': f'{w}%'})


def get_rankings_div(tot_content: int) -> html.Div:
    all_rankings = [html.Div(children=[
        dbc.Row(children=[
            dbc.Col(dbc.Label(f"Spaceship in {i + 1}{['st', 'nd', 'rd', 'th'][i] if i < 4 else 'th'} place:"),
                    width={'offset': 5, 'size': 1}),
            dbc.Col(dbc.DropdownMenu(label='A',
                                     children=[dbc.DropdownMenuItem(f'{chr(ord("A") + j)}',
                                                                    id={'type': "spaceship-ranking", 'index': (i * tot_content) + j}) for j in range(tot_content)],
                                     id={'type': "ranking-dropdown", 'index': i}),
                    width=1,
                    align='start')]),
        html.Br()]) for i in range(tot_content)]
    return html.Div(children=all_rankings,
                    id='ranking-div')


def set_app_layout():
    with open('./assets/help.md', 'r', encoding='utf-8') as f:
        info_str = f.read()
    info_modal = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Info"), close_button=True),
        dbc.ModalBody(dcc.Markdown(info_str))
    ],
                           id='info-modal',
                           centered=True,
                           backdrop='static',
                           is_open=False,
                           scrollable=True,
                           size='lg')
    
    err_modal = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("❌ Error ❌"), close_button=True),
        dbc.ModalBody(dcc.Markdown(scores_different_error))
    ],
                           id='err-modal',
                           centered=True,
                           backdrop='static',
                           is_open=False,
                           scrollable=True)
    
    ok_modal = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("✔️ Success ✔️"), close_button=True),
        dbc.ModalBody(dcc.Markdown(rankings_assigned))
    ],
                           id='ok-modal',
                           centered=True,
                           backdrop='static',
                           is_open=False,
                           scrollable=True)
        
    header = dbc.Row(children=[
                dbc.Col(html.H1(children='🚀Space Engineers Spaceships Ranker🚀',
                                className='title'), width={'size': 10, 'offset': 1}),
                dbc.Col(children=[dbc.Button('Info',
                                             id='info-btn',
                                             color='info')],
                        align='center', width=1)
    ],
                     className='header')
        
    upload_component = html.Div(children=[
        dcc.Upload(id='upload-data',
                   children=html.Div([
                       'Drag and Drop or ',
                       html.A('Select Files')
                       ]),
                   style={
                       'width': '60%',
                       'height': '60px',
                       'lineHeight': '60px',
                       'borderWidth': '1px',
                       'borderStyle': 'dashed',
                       'borderRadius': '5px',
                       'textAlign': 'center',
                       'margin': '10px auto'
                       },
                   # Allow multiple files to be uploaded
                   multiple=True
                   )])
    
    upload_progress = html.Div(
        children=[
            html.Br(),
            dbc.Label('Uploading progress: '),
            dbc.Progress(id="upload-progress",
                         color='info',
                         striped=True,
                         animated=True)
        ],
        id='upload-progress-div',
        style={'content-visibility': 'visible' if 0 <= progress <= 100 else 'hidden'})
    
    content_container = html.Div(children=[get_content_div(i, len(emitters)) for i in range(len(emitters))],
                                 style={'width': '100%', 'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center'})
    
    ranking_div = get_rankings_div(tot_content=len(emitters))
    
    save_data = dbc.Row(children=[
        dbc.Col(children=[dbc.Button(children='Save',
                                     id='save-btn',
                                     n_clicks=0,
                                     disabled=False,
                                     color="primary",
                                     size="lg",
                                     className="me-1"),
                          dcc.Download(id='save-data')],
                align='center', width={'offset': 6, 'size': 1})])
    
    app.layout = dbc.Container(
        children=[
            info_modal,
            err_modal,
            ok_modal,
            header,
            html.Br(),
            upload_component,
            html.Br(),
            upload_progress,
            html.Br(),
            content_container,
            html.Br(),
            ranking_div,
            html.Br(),
            save_data,
            dcc.Interval(id='interval',
                         interval=1 * 1000,
                         n_intervals=0),
        ],
        fluid=True)


@app.callback(
    Output("info-modal", "is_open"),
    Input("info-btn", "n_clicks"),
    prevent_initial_call=True
)
def show_webapp_info(n):
    return True


@app.callback(
    [Output("upload-progress", "value"),
     Output("upload-progress", "label"),
     Output('upload-progress-div', 'style')],
    [Input("interval", "n_intervals")],
)
def update_progress(n):  
    return progress, f"{progress}%", {'content-visibility': 'visible' if 0 <= progress <= 100 else 'hidden'}



@app.callback(
    Output({'type': 'ranking-dropdown', 'index': ALL}, "label"),
    Output({'type': 'spaceship-ranking', 'index': ALL}, "n_clicks"),
    State({'type': 'ranking-dropdown', 'index': ALL}, "label"),
    Input({'type': 'spaceship-ranking', 'index': ALL}, "n_clicks"),
    prevent_initial_call=True
)
def update_dropdown_value(labels: List[str],
                          vs: List[Optional[int]]):
    # super hacky but Dash's pattern matching fails in this case
    label_changed = vs.index(1) // len(labels)
    new_label = chr(ord("A") + (vs.index(1) % len(labels)))
    labels[label_changed] = new_label
    return labels, [None] * len(vs)


def parse_contents(filename,
                   contents):
    _, rngseed, exp_n = filename.split('_')
    rngseed = int(rngseed)
    exp_n = int(exp_n.replace('exp', ''))

    _, content_string = contents.split(',')
    cs_string = base64.b64decode(content_string).decode(encoding='utf-8')

    return rngseed, exp_n, cs_string


def get_content_plot(spaceship: CandidateSolution) -> go.Figure:
    global lsystem
    global hull_builder
    
    spaceship = lsystem._set_structure(cs=lsystem._add_ll_strings(cs=spaceship))
    hull_builder.add_external_hull(structure=spaceship.content)
    content = spaceship.content.as_grid_array
    arr = np.nonzero(content)
    x, y, z = arr
    cs = [content[i, j, k] for i, j, k in zip(x, y, z)]
    ss = [spaceship.content._clean_label(list(block_definitions.keys())[v - 1]) for v in cs]
    fig = px.scatter_3d(x=x,
                        y=y,
                        z=z,
                        color=ss,
                        color_discrete_map=_get_colour_mapping(ss),
                        labels={
                            'x': '',
                            'y': 'm',
                            'z': '',
                            'color': 'Block type'
                        },
                        title='',
                        template='plotly_dark')
    
    ux, uy, uz = np.unique(x), np.unique(y), np.unique(z)
    ptg = .2
    show_x = [v for i, v in enumerate(ux) if i % (1 / ptg) == 0]
    show_y = [v for i, v in enumerate(uy) if i % (1 / ptg) == 0]
    show_z = [v for i, v in enumerate(uz) if i % (1 / ptg) == 0]
    
    fig.update_layout(
        scene=dict(
            xaxis={
                'tickmode': 'array',
                'tickvals': show_x,
                'ticktext': [spaceship.content.grid_size * i for i in show_x],
            },
            yaxis={
                'tickmode': 'array',
                'tickvals': show_y,
                'ticktext': [spaceship.content.grid_size * i for i in show_y],
            },
            zaxis={
                'tickmode': 'array',
                'tickvals': show_z,
                'ticktext': [spaceship.content.grid_size * i for i in show_z],
            }
        )
    )
    
    fig.update_traces(marker=dict(size=4,
                              line=dict(width=3,
                                        color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=2, y=2, z=2)
        )
    fig.update_layout(scene=dict(aspectmode='data'),
                      scene_camera=camera,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      showlegend=False)
    return fig


@app.callback(
    Output("save-data", "data"),
    Output('err-modal', "is_open"),
    Output('ok-modal', "is_open"),

    Input("save-btn", "n_clicks"),

    State({'type': 'ranking-dropdown', 'index': ALL}, "label"),
    prevent_initial_call=True
)
def download_scores(n_clicks,
                    rankings):
    global rng_seed

    random.seed(rng_seed)
    my_emitterslist = emitters.copy()
    random.shuffle(my_emitterslist)
    res = {emitter: ord(v) - ord('A') + 1 for emitter, v in zip(my_emitterslist, rankings)}
    if len(set(rankings)) == len(rankings):
        return dict(content=str(res), filename=f'{str(rng_seed)}_res.json'), False, True
    else:
        return None, True, False


@app.callback(
    Output({'type': 'spaceship-content', 'index': ALL}, 'figure'),
    Output('upload-data', 'contents'),
    Output('upload-data', 'filename'),
    
    Input('upload-data', 'contents'),

    State('upload-data', 'filename'),
    State({'type': 'spaceship-content', 'index': ALL}, 'figure'),
)
def general_callback(list_of_contents, list_of_names, spaceship_plot):
    global rng_seed
    global progress

    ctx = dash.callback_context

    if not ctx.triggered:
        event_trig = None
    else:
        event_trig = ctx.triggered[0]['prop_id'].split('.')[0]

    if event_trig == 'upload-data':
        children = [parse_contents(n, c) for c, n in zip(list_of_contents, list_of_names)]
        progress = 0
        for child in children:
            progress += 100 / len(children)
            rng_seed, exp_n, cs_string = child
            cs = CandidateSolution(string=cs_string)
            spaceship_plot[exp_n]  = get_content_plot(spaceship=cs)
        progress = -1

    return spaceship_plot, '', ''
