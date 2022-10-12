import base64
import json
import logging
import os
import random
import sys
from turtle import width
from typing import Dict, List, Optional, Tuple

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, dcc, html
from dash.dependencies import Input, Output, State
from pcgsepy.common.api_call import block_definitions
from pcgsepy.common.vecs import Vec
from pcgsepy.evo.genops import expander
from pcgsepy.guis.ships_comparator.modals_msgs import (rankings_assigned,
                                                       scores_different_error)
from pcgsepy.hullbuilder import HullBuilder
from pcgsepy.lsystem.solution import CandidateSolution
from pcgsepy.setup_utils import get_default_lsystem


def resource_path(relative_path):
    # get absolute path to resource
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


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
                           apply_smoothing=False)

emitters = ['Human', 'Random', 'Greedy', 'Contextual Bandit']
progress = -1
base_color = Vec.v3f(0.5, 0.5, 0.5)
rng_seed = None

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
                title='Spaceships Ranker',
                external_stylesheets=[dbc.themes.DARKLY],
                assets_folder=resource_path("assets"),
                update_title=None)


def is_base_block(block_type: str) -> bool:
    """Check if the block is a base block. Base blocks are non-functional, structural blocks.

    Args:
        block_type (str): The type of the block.

    Returns:
        bool: Whether the block is a base block.
    """
    return block_type.endswith("Block") or block_type.endswith("Slope") or block_type.endswith("Corner") or block_type.endswith("CornerInv")


def get_content_div(content_n: int) -> html.Div:
    """Generate the `html.Div` that contains the content plot.

    Args:
        content_n (int): The number of the content.

    Returns:
        html.Div: The container for the content plot.
    """
    w = 100 // len(emitters)
    return html.Div(children=[
        # title
        html.Div(children=[
            html.H1(children=f'Spaceship {chr(ord("A") + content_n)}',
                    style={'text-align': 'center'})
        ]),
        html.Br(),
        # content display
        html.Div(children=[
            dcc.Graph(id={'type': "spaceship-content", 'index': content_n},
                      figure=px.scatter_3d(x=np.zeros(0, dtype=object),
                                           y=np.zeros(0, dtype=object),
                                           z=np.zeros(0, dtype=object),
                                           title='',
                                           template='plotly_dark'),
                      config={'displayModeBar': False,
                              'displaylogo': False})],
                 className='content-div',
                 style={'width': '100%'})],
                    style={'width': f'{w}%'})


def get_rankings_div() -> html.Div:
    """Generate the `html.Div` for the rankings assignment.

    Returns:
        html.Div: The container for the rankings.
    """
    rankings_div = html.Div(children=[
        dbc.Row(children=[
            dbc.Col(children=[html.H1("Spaceships Ranking")],
                    style={'text-align': 'center'},
                    width={'offset': 4, 'size': 4})]),            
        html.Br(),
        ],
                            id='ranking-div')    
    rankings_div.children.append(
        dbc.Row(children=[
            dbc.Col(children=[
                dbc.Row(
                    dbc.Col(dbc.Label(f"{i + 1}{['st', 'nd', 'rd', 'th'][i] if i < 4 else 'th'} place:", style={'font-size': 'x-large'})),
                    ),
                dbc.Row(
                    dbc.Col(dbc.DropdownMenu(label='A',
                                             style={'font-size': 'x-large'},
                                             children=[dbc.DropdownMenuItem(f'{chr(ord("A") + j)}',
                                                                            id={'type': "spaceship-ranking", 'index': (i * len(emitters)) + j}) for j in range(len(emitters))],
                                             id={'type': "ranking-dropdown", 'index': i}))
                    )
                ],
                    style={'text-align': 'center'},
                    width={'offset': 4 if i == 0 else 0, 'size': 1}) for i in range(len(emitters))
            ])
        )
            
    
    return rankings_div


def parse_contents(filename: str,
                   contents: str) -> Tuple[int, int, str, Vec]:
    """Parse the contents of the uploaded data.

    Args:
        filename (str): The name of the file.
        contents (str): The base-64 encoded contents of the file.

    Returns:
        Tuple[int, int, str]: The RNG seed, the number of the experiment, and the content string.
    """
    _, rngseed, exp_n = filename.split('_')
    rngseed = int(rngseed)
    exp_n = int(exp_n.replace('exp', ''))
    _, cs_properties = contents.split(',')
    cs_properties = json.loads(base64.b64decode(cs_properties).decode(encoding='utf-8'))
    cs_string, base_color = cs_properties['string'], Vec.from_json(cs_properties['base_color'])
    logging.getLogger('webapp').debug(msg=f'[{__name__}.parse_contents] {rngseed=}; {exp_n=}; {cs_string=}; {base_color=}')
    return rngseed, exp_n, cs_string, base_color


def get_content_plot(spaceship: CandidateSolution) -> go.Figure:
    """Generate the content plot figure.

    Args:
        spaceship (CandidateSolution): The spaceship to plot.

    Returns:
        go.Figure: The content plot.
    """
    spaceship = lsystem._set_structure(cs=lsystem._add_ll_strings(cs=spaceship))
    hull_builder.add_external_hull(structure=spaceship.content)
    content = spaceship.content.as_grid_array
    arr = np.nonzero(content)
    x, y, z = arr
    cs = [content[i, j, k] for i, j, k in zip(x, y, z)]
    ss = [spaceship.content._clean_label(list(block_definitions.keys())[v - 1]) for v in cs]
    custom_colors = []
    for (i, j, k) in zip(x, y, z):
        b = spaceship.content._blocks[(i * spaceship.content.grid_size, j * spaceship.content.grid_size, k * spaceship.content.grid_size)]
        if is_base_block(block_type=b.block_type):
            custom_colors.append(f'rgb{base_color.as_tuple()}')
        else:
            custom_colors.append(block_to_colour.get(spaceship.content._clean_label(b.block_type), '#ff0000'))
    
    fig = go.Figure()
    
    fig.add_scatter3d(x=x,
                        y=y,
                        z=z,
                        mode='markers',
                        marker=dict(size=4,
                                    line=dict(width=3,
                                            color='DarkSlateGrey'),
                                    color=custom_colors),
                        showlegend=False
                        )
    
    fig.update_traces(
        hoverinfo='text',
        hovertext=ss
    )
    
    ux, uy, uz = np.unique(x), np.unique(y), np.unique(z)
    ptg = .2
    show_x = [v for i, v in enumerate(ux) if i % (1 / ptg) == 0]
    show_y = [v for i, v in enumerate(uy) if i % (1 / ptg) == 0]
    show_z = [v for i, v in enumerate(uz) if i % (1 / ptg) == 0]
    
    fig.update_layout(
        scene=dict(
            xaxis_title='',
            yaxis_title='m',
            zaxis_title='',
            xaxis={
                'tickmode': 'array',
                'tickvals': show_x,
                'ticktext': [spaceship.content.grid_size * i for i in show_x],
            },
            yaxis={
                # 'tickmode': 'array',
                'tickvals': show_y,
                'ticktext': [spaceship.content.grid_size * i for i in show_y],
            },
            zaxis={
                # 'tickmode': 'array',
                'tickvals': show_z,
                'ticktext': [spaceship.content.grid_size * i for i in show_z],
            }
        )
    )
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=2, y=2, z=2)
        )
    
    fig.update_layout(scene=dict(aspectmode='data'),
                      scene_camera=camera,
                      template='plotly_dark',
                    #   paper_bgcolor='rgba(0,0,0,0)',
                    #   plot_bgcolor='rgba(0,0,0,0)',
                    margin=go.layout.Margin(
                          l=0,
                          r=0,
                          b=0,
                          t=0)
                      )
    return fig


def set_app_layout():
    """Load resources and set the layout of the web application."""
    # load resource files
    with open('./assets/help.md', 'r', encoding='utf-8') as f:
        info_str = f.read()
    # create modals
    info_modal = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Info"),
                        style={'flex-direction': 'column-reverse'},
                        close_button=True),
        dbc.ModalBody(dcc.Markdown(info_str),
                      style={'text-align': 'justify'})
    ],
        id='info-modal',
        centered=True,
        backdrop='static',
        is_open=False,
        scrollable=True,
        size='lg')
    err_modal = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("❌ Error ❌"),
                        style={'flex-direction': 'column-reverse'}, 
                        close_button=False),
        dbc.ModalBody(dcc.Markdown(scores_different_error))
    ],
        id='err-modal',
        centered=True,
        backdrop=True,
        is_open=False,
        scrollable=True)
    ok_modal = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("✔️ Success ✔️"),
                        style={'flex-direction': 'column-reverse'}, 
                        close_button=False),
        dbc.ModalBody(dcc.Markdown(rankings_assigned))
    ],
        id='ok-modal',
        centered=True,
        backdrop=True,
        is_open=False,
        scrollable=True)
    # create containers
    header = dbc.Row(children=[
        dbc.Col(html.H1(children='🚀Space Engineers Spaceships Ranker🚀',
                        className='title'), width={'size': 6, 'offset': 3}),
        dbc.Col(children=[dbc.Button('Info',
                                     id='info-btn',
                                     color='info')],
                align='center', width=1)
    ],
        className='header')
    upload_component = html.Div(children=[
        dcc.Upload(id='upload-data',
                   children=html.Div(children=[
                       'Drag and Drop or ',
                       html.A('Select Files')
                   ],
                                     style={'font-size': 'x-large'}),
                   style={
                       'width': '60%',
                       'height': '60px',
                       'lineHeight': '60px',
                       'margin': '10px auto'
                   },
                   className='upload',
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
    content_container = html.Div(children=[get_content_div(i) for i in range(len(emitters))],
                                 style={'width': '100%', 'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center'})
    save_data = dbc.Row(children=[
        dbc.Col(children=[dbc.Button(children='Save',
                                     style={'font-size': 'xx-large'},
                                     id='save-btn',
                                     n_clicks=0,
                                     disabled=True,
                                     color="primary",
                                     size="lg",
                                     className="me-1"),
                          dcc.Download(id='save-data')],
                align='center',
                width={'offset': 4, 'size': 4},
                style={'text-align': 'center'})])
    # set the app layout
    app.layout = dbc.Container(
        children=[
            info_modal,
            err_modal,
            ok_modal,
            header,
            html.Br(),
            dbc.Row(
                dbc.Col(
                    [
                        upload_component,
                        html.Br(),
                        upload_progress,
                        ],
                    width={'size': 10, 'offset': 1}
                )
            ),
            html.Br(),
            dbc.Row(
                dbc.Col(
                    [
                        content_container,
                        html.Br(),
                        dbc.Row(
                            dbc.Col(
                                get_rankings_div(),
                                width=('offset: 1')
                            )
                        ),
                        html.Br(),
                        html.Br(),
                        save_data
                        ],
                    width={'size': 10, 'offset': 1})
            ),
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


@app.callback(
    Output("save-data", "data"),
    Output('err-modal', "is_open"),
    Output('ok-modal', "is_open"),

    Input("save-btn", "n_clicks"),

    State({'type': 'ranking-dropdown', 'index': ALL}, "label"),
    prevent_initial_call=True
)
def download_scores(n_clicks: int,
                    rankings: List[str]) -> Tuple[Optional[Dict[str, str]], bool, bool]:
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
    Output('save-btn', 'disabled'),

    Input('upload-data', 'contents'),

    State('upload-data', 'filename'),
    State({'type': 'spaceship-content', 'index': ALL}, 'figure'),
)
def general_callback(list_of_contents: List[str],
                     list_of_names: List[str],
                     spaceship_plot: List[go.Figure]) -> Tuple[List[go.Figure], str, str]:
    global rng_seed
    global progress
    global base_color
    
    savebtn_disabled = True

    ctx = dash.callback_context

    if not ctx.triggered:
        event_trig = None
    else:
        event_trig = ctx.triggered[0]['prop_id'].split('.')[0]

    logging.getLogger('webapp').debug(msg=f'[{__name__}.general_callback] {event_trig=}')
    
    if event_trig == 'upload-data':
        children = [parse_contents(n, c) for c, n in zip(list_of_contents, list_of_names)]
        progress = 0
        for child in children:
            progress += 100 / len(children)
            rng_seed, exp_n, cs_string, cs_base_color = child
            base_color = cs_base_color
            cs = CandidateSolution(string=cs_string)
            spaceship_plot[exp_n - 1] = get_content_plot(spaceship=cs)
        progress = -1
        savebtn_disabled = False

    return spaceship_plot, '', '', savebtn_disabled
