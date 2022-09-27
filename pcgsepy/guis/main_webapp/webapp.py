import base64
from cProfile import run
import json
import logging
import os
import random
from re import A
import sys
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zipfile import ZipFile

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    os.chdir(sys._MEIPASS)

import dash
import dash_bootstrap_components as dbc
import dash_daq as daq
import numpy as np
import plotly.graph_objects as go
from dash import ALL, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from pcgsepy.common.api_call import block_definitions
from pcgsepy.common.jsonifier import json_dumps
from pcgsepy.common.vecs import Vec
from pcgsepy.config import (BIN_POP_SIZE, CS_MAX_AGE, MY_EMITTERS, N_EMITTER_STEPS,
                            N_GENS_ALLOWED)
from pcgsepy.guis.main_webapp.modals_msgs import (end_of_experiment,
                                                  end_of_userstudy,
                                                  no_selection_error,
                                                  privacy_policy_body,
                                                  privacy_policy_question)
from pcgsepy.hullbuilder import HullBuilder
from pcgsepy.lsystem.rules import StochasticRules
from pcgsepy.lsystem.solution import CandidateSolution
from pcgsepy.mapelites.behaviors import (BehaviorCharacterization, avg_ma,
                                         mame, mami, symmetry)
from pcgsepy.mapelites.bin import MAPBin
from pcgsepy.mapelites.emitters import (ContextualBanditEmitter, Emitter,
                                        GreedyEmitter, HumanEmitter,
                                        HumanPrefMatrixEmitter,
                                        PreferenceBanditEmitter, RandomEmitter)
from pcgsepy.mapelites.map import MAPElites, get_elite
from pcgsepy.xml_conversion import convert_structure_to_xml
from tqdm import trange


class DashLoggerHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)
        self.queue = []

    def emit(self, record):
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = self.format(record)
        self.queue.append(f'[{t}]\t{msg}')


dashLoggerHandler = DashLoggerHandler()
logging.getLogger('webapp').addHandler(dashLoggerHandler)


base_color: Vec = Vec.v3f(0.45, 0.45, 0.45)
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
consent_ok: Optional[bool] = None
current_mapelites: Optional[MAPElites] = None
exp_n: int = 0
gen_counter: int = 0
gdev_mode: bool = False
hidden_style = {'visibility': 'hidden', 'height': '0px', 'display': 'none'}
hm_callback_props = {}
my_emitterslist: List[str] = MY_EMITTERS.copy()
behavior_descriptors = [
    BehaviorCharacterization(name='Major axis / Medium axis',
                             func=mame,
                             bounds=(0, 10)),
    BehaviorCharacterization(name='Major axis / Smallest axis',
                             func=mami,
                             bounds=(0, 20)),
    BehaviorCharacterization(name='Average Proportions',
                             func=avg_ma,
                             bounds=(0, 20)),
    BehaviorCharacterization(name='Symmetry',
                             func=symmetry,
                             bounds=(0, 1))
]
rngseed: int = 42
selected_bins: List[Tuple[int, int]] = []
step_progress: int = -1
use_custom_colors = True
user_study_mode: bool = True


def resource_path(relative_path):
# get absolute path to resource
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class Metric:
    def __init__(self,
                 multiple_values: bool = False) -> None:
        self.current_generation: int = 0
        self.multiple_values = multiple_values
        self.history: Dict[int, List[Any]] = {
            self.current_generation: [] if multiple_values else 0
        }
        self.emitter_names: List[str] = [my_emitterslist[exp_n]]
    
    def add(self,
            value: Any):
        if self.multiple_values:
            self.history[self.current_generation].append(value)
        else:
            self.history[self.current_generation] += value
    
    def reset(self):
        if self.multiple_values:
            self.history[self.current_generation] = []
        else:
            self.history[self.current_generation] = 0
    
    def new_generation(self):
        self.current_generation += 1
        self.reset()
        self.emitter_names.append(my_emitterslist[exp_n])
    
    def get_averages(self) -> List[Any]:
        return [np.mean(l) for l in self.history.values()]


n_spaceships_inspected = Metric()
time_elapsed = Metric(multiple_values=True)


class Semaphore:
    def __init__(self,
                 locked: bool = False) -> None:
        self._is_locked = locked
        self._running = ''
    
    @property
    def is_locked(self) -> bool:
        return self._is_locked
    
    def lock(self,
             name: Optional[str] = ''):
        self._is_locked = True
        self._running = name
    
    def unlock(self):
        self._is_locked = False
        self._running = ''

download_semaphore = Semaphore(locked=True)    
process_semaphore = Semaphore()


app = dash.Dash(__name__,
                title='AI Spaceship Generator',
                external_stylesheets=[dbc.themes.DARKLY],
                assets_folder=resource_path("assets"),
                update_title=None)


def set_callback_props(mapelites: MAPElites):
    hm_callback_props['pop'] = {
        'Feasible': 'feasible',
        'Infeasible': 'infeasible'
    }
    hm_callback_props['metric'] = {
        'Fitness': {
            'name': 'fitness',
            'zmax': {
                'feasible': sum([x.weight * x.bounds[1] for x in mapelites.feasible_fitnesses]) + mapelites.nsc,
                'infeasible': 1.
            },
            'colorscale': 'Inferno'
        },
        'Age':  {
            'name': 'age',
            'zmax': {
                'feasible': CS_MAX_AGE,
                'infeasible': CS_MAX_AGE
            },
            'colorscale': 'Greys'
        },
        'Coverage': {
            'name': 'size',
            'zmax': {
                'feasible': BIN_POP_SIZE,
                'infeasible': BIN_POP_SIZE
            },
            'colorscale': 'Hot'
        }
    }
    hm_callback_props['method'] = {
        'Population': True,
        'Elite': False
    }


def get_properties_table(cs: Optional[CandidateSolution] = None) -> dbc.Table:
    size = str(cs.size) if cs else '-' 
    nblocks = cs.n_blocks if cs else '-'
    vol = cs.content.total_volume if cs else '-'
    mass = cs.content.mass if cs else '-'
    
    table_header = [
        html.Thead(html.Tr([html.Th("Property", style={'text-align': 'center'}),
                            html.Th("Value", style={'text-align': 'center'})]))
        ]
    table_body = [html.Tbody([
        html.Tr([html.Td("Spaceship size"), html.Td(f'{size} m', style={'text-align': 'center'})]),
        html.Tr([html.Td("Number of blocks"), html.Td(nblocks, style={'text-align': 'center'})]),
        html.Tr([html.Td("Occupied volume"), html.Td(f'{vol} m³', style={'text-align': 'center'})]),
        html.Tr([html.Td("Spaceship mass"), html.Td(f'{mass} kg', style={'text-align': 'center'})]),
    ])]
    
    return table_header + table_body


def _get_emitter() -> Emitter:
    curr_emitter = my_emitterslist[exp_n].replace('.json', '').split('_')[1]
    if curr_emitter == 'human':
        return HumanEmitter()
    elif curr_emitter == 'random':
        return RandomEmitter()
    elif curr_emitter == 'greedy':
        return GreedyEmitter()
    elif curr_emitter == 'contbandit':
        return ContextualBanditEmitter()
    else:
        raise ValueError(f'Unexpected emitter type: {curr_emitter} (from "{my_emitterslist[exp_n]}"')
   
 
def set_app_layout(mapelites: Optional[MAPElites] = None,
                   dev_mode: bool = True):
    
    global current_mapelites
    global rngseed
    global consent_ok
    global gdev_mode
    global user_study_mode
    
    gdev_mode = dev_mode
    user_study_mode = not gdev_mode
    consent_ok = False if gdev_mode else None
    
    webapp_info_file = './assets/webapp_help_dev.md' if dev_mode else './assets/webapp_info.md'
    with open(webapp_info_file, 'r', encoding='utf-8') as f:
        webapp_info_str = f.read()
        
    algo_info_file = './assets/algo_info.md'
    with open(algo_info_file, 'r', encoding='utf-8') as f:
        algo_info_str = f.read()
    
    quickstart_info_file = './assets/quickstart.md'
    with open(quickstart_info_file, 'r', encoding='utf-8') as f:
        quickstart_info_str = f.read()
    
    quickstart_usermode_info_file = './assets/quickstart_usermode.md'
    with open(quickstart_usermode_info_file, 'r', encoding='utf-8') as f:
        quickstart_usermode_info_str = f.read()
    
    current_mapelites = mapelites
    
    rngseed = uuid.uuid4().int
    if not gdev_mode:
        random.seed(rngseed)
        random.shuffle(my_emitterslist)
        current_mapelites.emitter = _get_emitter()
    
    logging.getLogger('webapp').info(msg=f'Your ID is {rngseed}.')
    
    logging.getLogger('webapp').debug(msg=f'[{__name__}.set_app_layout] {rngseed=}; {my_emitterslist=}.')
    
    
    consent_dialog = dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Privacy Policy"),
                            style={'justify-content': 'center'},
                            close_button=False),
            dbc.ModalBody(children=[dcc.Markdown(privacy_policy_body,
                                                 link_target="_blank",
                                                 style={'text-align': 'justify'}),
                                    dcc.Markdown(privacy_policy_question,
                                                 style={'text-align': 'center'})
                                    ]),
            dbc.ModalFooter(children=[
                dbc.Button("No",
                           disabled=False,
                           id="consent-no",
                           color="danger",
                           className="ms-auto",
                           n_clicks=0,
                           style={'width': '49%'}),
                dbc.Button("Yes",
                           disabled=False,
                           id="consent-yes",
                           color="success",
                           className="ms-auto",
                           n_clicks=0,
                           style={'width': '49%'})
                ])
            ],
        id="consent-modal",
        centered=True,
        backdrop="static",
        keyboard=False,
        is_open=False)
    
    webapp_info_modal = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Webapp Info"),
                        style={'flex-direction': 'column-reverse'},
                        close_button=True),
        dbc.ModalBody(dcc.Markdown(webapp_info_str,
                                   style={'text-align': 'justify'}))
        ],
                           id='webapp-info-modal',
                           centered=True,
                           backdrop='static',
                           is_open=False,
                           scrollable=True,
                           size='lg')
    
    algo_info_modal = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("AI Info"),
                        style={'flex-direction': 'column-reverse'},
                        close_button=True),
        dbc.ModalBody(dcc.Markdown(algo_info_str,
                                   style={'text-align': 'justify'},
                                   mathjax=True))
    ],
                           id='algo-info-modal',
                           centered=True,
                           backdrop='static',
                           is_open=False,
                           scrollable=True,
                           size='lg')
    
    quickstart_modal = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Quickstart"),
                        style={'flex-direction': 'column-reverse'},
                        close_button=True),
        dbc.ModalBody(dcc.Markdown(quickstart_info_str,
                                   link_target="_blank",
                                   style={'text-align': 'justify'}))
    ],
                           id='quickstart-modal',
                           centered=True,
                           backdrop='static',
                           is_open=False,
                           scrollable=True,
                           size='lg')
    
    quickstart_usermode_modal = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Quickstart"),
                        style={'flex-direction': 'column-reverse'},
                        close_button=True),
        dbc.ModalBody(dcc.Markdown(quickstart_usermode_info_str,
                                   link_target="_blank",
                                   style={'text-align': 'justify'}))
    ],
                           id='quickstart-usermode-modal',
                           centered=True,
                           backdrop='static',
                           is_open=False,
                           scrollable=True,
                           size='lg')

    no_bins_selected_modal = dbc.Modal(children=[
        dbc.ModalHeader(dbc.ModalTitle("⚠ Warning ⚠"),
                        style={'justify-content': 'center'},
                        close_button=False),
        dbc.ModalBody(no_selection_error),
        dbc.ModalFooter(children=[dbc.Button("Ok",
                                             id="nbs-err-btn",
                                             color="primary",
                                             className="ms-auto",
                                             n_clicks=0)]),
        ],
                                       id='nbs-err-modal',
                                       centered=True,
                                       backdrop=True,
                                       is_open=False)
    
    end_of_experiment_modal = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("End of Generation"),
                        style={'justify-content': 'center'},
                        close_button=False),
        dbc.ModalBody(dcc.Markdown(end_of_experiment,
                                   style={'text-align': 'justify'}))
    ],
                           id='eoe-modal',
                           centered=True,
                           backdrop=True,
                           is_open=False,
                           scrollable=True)
    
    end_of_userstudy_modal = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("End of User Study"),
                        style={'justify-content': 'center'},
                        close_button=False),
        dbc.ModalBody(dcc.Markdown(end_of_userstudy,
                                   style={'text-align': 'justify'}))
    ],
                           id='eous-modal',
                           centered=True,
                           backdrop=True,
                           is_open=False,
                           scrollable=True)
    
    modals = html.Div(children=[
        consent_dialog, webapp_info_modal, algo_info_modal, quickstart_modal, quickstart_usermode_modal,
        no_bins_selected_modal, end_of_experiment_modal, end_of_userstudy_modal
    ])
    
    header = dbc.Row(children=[
                dbc.Col(html.H1(children='🚀Space Engineers AI Spaceship Generator🚀',
                                className='title'), width={'size': 6, 'offset': 3}),
                dbc.Col(children=[dbc.Button('Webapp Info',
                                             id='webapp-info-btn',
                                             color='info')],
                        align='center', width=1),
                dbc.Col(children=[dbc.Button('AI Info',
                                             id='ai-info-btn',
                                             color='info')],
                        align='center', width=1)
    ],
                     className='header')
    
    exp_progress = html.Div(
        id='study-progress-div',
        children=[
            dbc.Row(
                dbc.Col(
                    children=[
                        html.H4('Study Progress',
                                className='section-title',
                                style=hidden_style if gdev_mode else {}),
                        html.Br()],
                    width={'size': 12, 'offset': 0},
                    style={'text-align': 'center'}
                ),
                align='center'
            ),
            
            dbc.Row(
                dbc.Col(children=[
                    dbc.Label(f'Current Iteration',
                              style={'font-size': 'large'}),
                dbc.Progress(id="gen-progress",
                             color='success',
                             striped=False,
                             animated=False)
                ],
                        width={'size': 12, 'offset': 0},
                        style={'text-align': 'center'},
                        align='center')
            ),
            
            dbc.Row(
                dbc.Col(children=[
                    dbc.Label(f'Spaceships Generation Progress',
                              style={'font-size': 'large'}),
                    dbc.Progress(id="exp-progress",
                                 color='success',
                                 striped=False,
                                 animated=False)
                ],
                        width={'size': 12, 'offset': 0},
                        style={**{'text-align': 'center'}, **hidden_style} if gdev_mode else {'text-align': 'center'},
                        align='center',
                        id='exp-progress-div')
            )
        ],
        style=hidden_style if not user_study_mode and not gdev_mode else {}
        )
    
    mapelites_heatmap = html.Div(children=[
        html.H4('Spaceship Population',
                className='section-title'),
        html.Br(),
        html.Div(className='container',
                 children=[
                     dcc.Graph(id="heatmap-plot",
                               figure=go.Figure(data=[]),
                               config={
                                   'displayModeBar': False,
                                   'displaylogo': False,
                                   'scrollZoom': True},
                               className='content',
                               style={'z-index': 0}),
                     html.Div(id='heatmap-plot-container',
                              className='overlay',
                              style={'visibility': 'hidden',
                                     'display': 'none',
                                     'pointer-events': 'auto',
                                     'z-index': 1}),
                 ])
        ])
    
    mapelites_controls = html.Div(
        children=[
            html.H4(children='Plot Settings',
                    className='section-title'),
            html.Br(),
            dbc.Label('Choose which population to display.'),
            dbc.DropdownMenu(label='Feasible',
                            children=[
                                dbc.DropdownMenuItem('Feasible', id='population-feasible'),
                                dbc.DropdownMenuItem('Infeasible', id='population-infeasible'),
                            ],
                            id='population-dropdown'),
            html.Br(),
            dbc.Label('Choose which metric to plot.'),
            dbc.DropdownMenu(label='Fitness',
                            children=[
                                dbc.DropdownMenuItem('Fitness', id='metric-fitness'),
                                dbc.DropdownMenuItem('Age', id='metric-age'),
                                dbc.DropdownMenuItem('Coverage', id='metric-coverage'),
                            ],
                            id='metric-dropdown'),
            html.Br(),
            dbc.Label('Choose whether to compute the metric for the entire bin population or just the elite.'),
            dbc.RadioItems(id='method-radio',
                        options=[
                            {'label': 'Population', 'value': 'Population'},
                            {'label': 'Elite', 'value': 'Elite'}
                        ],
                        value='Population')
            ],
        style=hidden_style if not gdev_mode else {})
    
    content_plot = html.Div(children=[
        html.H4('Selected Spaceship',
                className='section-title'),
        html.Br(),
        dcc.Graph(id="content-plot",
                  figure=go.Figure(data=[]),
                  config={
                      'displayModeBar': False,
                      'displaylogo': False}),
        ])
    
    color_and_download = html.Div(
        children=[
            dbc.Row(
                dbc.Col([
                    dbc.Label("Spaceship Color",
                              style={'font-size': 'large'}),
                    dbc.Input(type="color",
                              id="color-picker",
                              value="#737373",
                              size='lg')],
                        width={'size': 10, 'offset': 1},
                        style={'text-align': 'center'})
            ),
            html.Br(),
            dbc.Row(
                dbc.Col([
                    dbc.Label("Download Content",
                              style={'font-size': 'large'}),
                    dbc.Button('Download',
                               id='download-btn',
                               disabled=False),
                    dcc.Download(id='download-content'),
                    html.Br(),
                    html.Br(),
                    dcc.Loading(id='download-spinner',
                                children='\n\n',
                                fullscreen=False,
                                color='#eeeeee',
                                type='default',
                                style={'justify-content': 'center'})
                    ],
                        width={'size': 10, 'offset': 1},
                        style={'text-align': 'center'})
            )
        ]
    )
     
    content_properties = html.Div(
        children=[
            dbc.Row(children=[
                dbc.Col(children=[
                    dbc.Table(children=get_properties_table(),
                            id='spaceship-properties',
                            bordered=True,
                            dark=True,
                            hover=True,
                            responsive=True,
                            striped=True),
                    html.Div([
                        html.P(children='Content String: '),
                        dbc.Textarea(id='content-string',
                                    value='',
                                    contentEditable=False,
                                    disabled=True,
                                    class_name='content-string-area')
                        ],
                            style=hidden_style if not gdev_mode else {})
                    ])],
                    align='center')])    

    properties_panel = html.Div(
        children=[dbc.Row(dbc.Col([
            html.H4('Spaceship Properties',
                    className='section-title'),
            html.Br()
            ])),
                  dbc.Row(
                      [
                          dbc.Col(color_and_download,
                                  align='center'),
                          dbc.Col(content_properties)
                      ]
                  )]
    )
    
    experiment_settings = html.Div(
        children=[
            html.H4(children='Experiment Settings',
                    className='section-title'),
            html.Br(),
            html.Div(children=[
                html.P(children='Valid bins are: ',
                       id='valid-bins'),
                html.P(children=f'Selected bin(s): {selected_bins}',
                       id='selected-bin')
                ]),
            html.Br(),
            dbc.InputGroup(children=[
                dbc.InputGroupText('Feature Descriptors (X, Y):'),
                dbc.DropdownMenu(label=current_mapelites.b_descs[0].name,
                             children=[
                                 dbc.DropdownMenuItem(b.name, id=f"bc0-{b.name.replace(' / ', '_').replace(' ', '-')}")
                                 for b in behavior_descriptors],
                             id='b0-dropdown'),
                dbc.DropdownMenu(label=current_mapelites.b_descs[1].name,
                             children=[
                                 dbc.DropdownMenuItem(b.name, id=f"bc1-{b.name.replace(' / ', '_').replace(' ', '-')}")
                                 for b in behavior_descriptors],
                             id='b1-dropdown')
                ],
                           className="mb-3",
                           style=hidden_style if not gdev_mode else {}),
            dbc.InputGroup(children=[
                dbc.InputGroupText('Toggle L-system Modules:'),
                dbc.Checklist(id='lsystem-modules',
                              options=[{'label': x.name, 'value': x.name} for x in current_mapelites.lsystem.modules],
                              value=[x.name for x in current_mapelites.lsystem.modules if x.active],
                              inline=True,
                              switch=True)
                ],
                           style=hidden_style if not gdev_mode else {},
                           className="mb-3"),
            dbc.InputGroup(children=[
                dbc.InputGroupText('Fitness Weights:'),
                html.Div(children=[
                    html.Div(children=[
                        dbc.Label(children=f.name,
                                  style={'font-size': 'large'}),
                        html.Div(children=[
                            dcc.Slider(min=0,
                                       max=1,
                                       step=0.1,
                                       value=1,
                                       marks=None,
                                       tooltip={"placement": "bottom",
                                                "always_visible": False},
                                       id={'type': 'fitness-sldr',
                                           'index': i})
                        ],
                                 )
                        ]) for i, f in enumerate(current_mapelites.feasible_fitnesses)
                    ])
                ],
                           style={'content-visibility': 'hidden', 'visibility': 'hidden', 'height': '0px'} if not gdev_mode else {},
                           className="mb-3"),
            dbc.InputGroup(children=[
                dbc.InputGroupText('Select Emitter:'),
                dbc.DropdownMenu(label='Human',
                             children=[
                                 dbc.DropdownMenuItem('Human'),
                                 dbc.DropdownMenuItem('Random'),
                                 dbc.DropdownMenuItem('Greedy'),
                                 dbc.DropdownMenuItem('Preference Matrix'),
                                 dbc.DropdownMenuItem('Preference Bandit'),
                                 dbc.DropdownMenuItem('Contextual Bandit'),
                             ],
                             id='emitter-dropdown')
                ],
                           className="mb-3",
                           style=hidden_style if not gdev_mode else {}),
            dbc.InputGroup(children=[
                dbc.InputGroupText('Enforce Symmetry:'),
                dbc.DropdownMenu(label='None',
                             children=[
                                 dbc.DropdownMenuItem('None', id='symmetry-none'),
                                 dbc.DropdownMenuItem('X-axis', id='symmetry-x'),
                                 dbc.DropdownMenuItem('Y-axis', id='symmetry-y'),
                                 dbc.DropdownMenuItem('Z-axis', id='symmetry-z'),
                             ],
                             id='symmetry-dropdown',
                             style=hidden_style if not gdev_mode else {}),
                dbc.RadioItems(id='symmetry-radio',
                        options=[
                            {'label': 'Upper', 'value': 'Upper'},
                            {'label': 'Lower', 'value': 'Lower'}
                        ],
                        value='Upper')
                ],
                           style=hidden_style if not gdev_mode else {},
                           className="mb-3"),
            dbc.InputGroup(children=[
                dbc.InputGroupText('Save/Load Population:'),
                dbc.Button(id='popdownload-btn',
                           children='Download Current Population'),
                dcc.Upload(
                    id='popupload-data',
                    children='Upload Population',
                    multiple=False
                    ),
                ],
                           className="mb-3",
                           style=hidden_style if not gdev_mode else {})
            ],
        style=hidden_style if user_study_mode else {})
    
    experiment_controls = html.Div(
        children=[
            html.H4('Experiment Controls',
                    className='section-title'),
            html.Br(),
            dbc.Row(dbc.Col(children=[
                dbc.Button(id='step-btn',
                           children='Evolve From Selected Spaceship',
                           className='button-fullsize')
                ],
                    width={'size': 4, 'offset':4})),
            html.Br(),
            dbc.Row(dbc.Col(children=[
                dbc.Button(id='rand-step-btn',
                           children='Evolve From Random Spaceship',
                           className='button-fullsize')
                ],
                    id='rand-step-btn-div',
                    style=hidden_style if user_study_mode or gdev_mode else {},
                    width={'size': 4, 'offset':4})),
            html.Br(),
            dbc.Row(dbc.Col(children=[
                dbc.Button(id='selection-clr-btn',
                       children='Clear Selection',
                           className='button-fullsize')
                ],
                    style=hidden_style if not gdev_mode else {},
                    width={'size': 4, 'offset':4})),
            html.Br(),
            dbc.Row(dbc.Col(children=[
                dbc.Button(id='selection-btn',
                       children='Toggle Single Bin Selection',
                           className='button-fullsize')
                ],
                    style=hidden_style if not gdev_mode else {},
                    width={'size': 4, 'offset':4})),
            html.Br(),
            dbc.Row(dbc.Col(children=[
                dbc.Button(id='reset-btn',
                           children='Reinitialize Population',
                           className='button-fullsize')
                ],
                    id='reset-btn-div',
                    style=hidden_style if user_study_mode else {},
                    width={'size': 4, 'offset':4})),
            html.Br(),
            dbc.Row(dbc.Col(children=[
                dbc.Button(id='subdivide-btn',
                       children='Subdivide Selected Bin(s)',
                           className='button-fullsize')
                ],
                    style=hidden_style if not gdev_mode else {},
                    width={'size': 4, 'offset':4})),
            html.Br(),
            dbc.Row(dbc.Col(children=[
                dbc.Button(id='download-mapelites-btn',
                           children='Download MAP-Elites',
                           className='button-fullsize'),
                dcc.Download(id='download-mapelites')
                ],
                    style=hidden_style if not gdev_mode else {},
                    width={'size': 4, 'offset':4})),
        ])
    
    rules = html.Div(
        children=[
            html.H4(children='High-level Rules',
                    className='section-title'),
            html.Br(),
            dbc.Textarea(id='hl-rules',
                         value=str(current_mapelites.lsystem.hl_solver.parser.rules),
                         wrap=False,
                         className='rules-area'),
            dbc.Row(
                dbc.Col(dbc.Button(children='Update High-level Rules',
                                   id='update-rules-btn'),
                        width={'size': 4, 'offset':4}),
                align='center')
            ],
        style=hidden_style if not gdev_mode else {})
    
    progress = html.Div(
        children=[
            dbc.Row(
                dbc.Col(children=[
                    dbc.Label('Evolution Progress',
                              style={'font-size': 'large'}),
                    dbc.Progress(id="step-progress",
                                 color='info',
                                 striped=True,
                                 animated=True)
                ])
            )
        ],
        id='step-progress-div',
        style={'content-visibility': 'visible' if 0 <= step_progress <= 100 else 'hidden',
               'display': 'inline-block' if 0 <= step_progress <= 100 else 'none',
               'width': '100%'})
    
    log = html.Div(
        children=[
            dcc.Interval(id='interval1',
                         interval=1 * 1000,
                         n_intervals=0),
            html.H4(children='Log',
                    className='section-title'),
            html.Br(),
            dbc.Textarea(id='console-out',
                         value='',
                         wrap=False,
                         contentEditable=False,
                         disabled=True,
                         className='log-area'),
            dcc.Interval(id='interval2',
                         interval=5 * 10,
                         n_intervals=0),
            ])
    
    load_spinner = html.Div(children=[
        dbc.Row(
            dbc.Col(children=[
                dcc.Loading(id='step-spinner',
                            children='',
                            fullscreen=False,
                            color='#eeeeee',
                            type='circle')
            ],
                width={'size': 2, 'offset': 5}   )
        )
        
    ])
    
    app.layout = dbc.Container(
        children=[
            modals,
            header,
            load_spinner,
            html.Br(),
            html.Br(),
            dbc.Row(children=[
                dbc.Col(mapelites_heatmap, width={'size': 3, 'offset': 1}),
                dbc.Col(content_plot, width=4),
                dbc.Col(properties_panel, width=3)],
                    align="start"),
            html.Br(),
            html.Br(),
            html.Br(),
            dbc.Row(children=[
                dbc.Col(children=[mapelites_controls,
                                  exp_progress,
                                  html.Br(),
                                  progress],
                        width={'size': 3, 'offset': 1}),
                dbc.Col(children=[experiment_controls,
                                  experiment_settings],
                        width=4),
                dbc.Col(children=[rules,
                                  log
                                  ],
                        width=3)],
                    align="start"),
            
            dcc.Download(id='download-population'),
            dcc.Download(id='download-metrics'),
            
            html.Div(id='hidden-div',
                     children=[],
                     style=hidden_style)
            ],
        fluid=True)


# clientside callback to open the Google Forms questionnaire on a new page
app.clientside_callback(
    """
    function(clicks) {
        if (clicks) {
            window.open("https://forms.gle/gsuajDXUNocZvDzn9", "_blank");
            return 0;
        }
    }
    """,
    Output("hidden-div", "n_clicks"),  # super hacky but Dash leaves me no choice
    Input("consent-yes", "n_clicks"),
    prevent_initial_call=True
)


# clientside callback to autoscroll log textarea
app.clientside_callback(
    """
    function checkTextareaHeight() {
        var textarea = document.getElementById("console-out");
        if(textarea.selectionStart == textarea.selectionEnd) {
            textarea.scrollTop = textarea.scrollHeight;
        }
        return "";
    }
    """,
    Output("hidden-div", "title"),  # super hacky but Dash leaves me no choice
    Input("interval1", "n_intervals"),
    prevent_initial_call=True
)


@app.callback(
    Output("webapp-info-modal", "is_open"),
    Input("webapp-info-btn", "n_clicks"),
    prevent_initial_call=True
)
def show_webapp_info(n):
    return True


@app.callback(
    Output("algo-info-modal", "is_open"),
    Input("ai-info-btn", "n_clicks"),
    prevent_initial_call=True
)
def show_algo_info(n):
    return True


@app.callback(Output('console-out', 'value'),
              Input('interval1', 'n_intervals'),
              prevent_initial_call=True)
def update_output(n):
    return ('\n'.join(dashLoggerHandler.queue))


@app.callback(
    [Output("step-progress", "value"),
     Output("step-progress", "label"),
     Output('step-progress-div', 'style')],
    [Input("interval1", "n_intervals")],
    prevent_initial_call=True
)
def update_progress(n):  
    return step_progress, f"{np.round(step_progress, 2)}%", {'content-visibility': 'visible' if 0 <= step_progress <= 100 else 'hidden', 
                                                             'display': 'inline-block' if 0 <= step_progress <= 100 else 'none',
                                                             'width': '100%'}


@app.callback(
    [Output("gen-progress", "value"),
     Output("gen-progress", "label")],
    [Input("interval1", "n_intervals")],
    prevent_initial_call=True
)
def update_gen_progress(n):
    if user_study_mode:
        val = np.round(100 * ((gen_counter) / N_GENS_ALLOWED), 2)
        return val, f"{gen_counter} / {N_GENS_ALLOWED}"
    else:
        return 100, f"{gen_counter}"


@app.callback(
    [Output("exp-progress", "value"),
     Output("exp-progress", "label")],
    [Input("interval1", "n_intervals")],
    prevent_initial_call=True
)
def update_exp_progress(n):
    val = min(100, np.round(100 * ((1 + exp_n) / len(my_emitterslist)), 2))
    return val, f"{min(exp_n + 1, len(my_emitterslist))} / {len(my_emitterslist)}"


@app.callback(
    Output("download-mapelites", "data"),
    Input("download-mapelites-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_mapelites(n_clicks):
    global current_mapelites
    global gen_counter
    
    t = datetime.now().strftime("%Y%m%d%H%M%S")
    fname = f'{t}_mapelites_{current_mapelites.emitter.name}_gen{str(gen_counter).zfill(2)}'
    logging.getLogger('webapp').info(f'The MAP-Elites object will be downloaded shortly.')
    return dict(content=json_dumps(current_mapelites), filename=f'{fname}.json')


@app.callback(
    Output("download-content", "data"),
    Output('download-spinner', 'children'),
    Input("download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_content(n):
    global selected_bins
    global current_mapelites
    global base_color
    global download_semaphore
    
    def write_archive(bytes_io):
        with ZipFile(bytes_io, mode="w") as zf:
            # with open('./assets/thumb.png', 'rb') as f:
            #     thumbnail_img = f.read()
            curr_content = _get_elite_content(mapelites=current_mapelites,
                                              bin_idx=_switch([selected_bins[-1]])[0],
                                              pop='feasible')
            thumbnail_img = curr_content.to_image(format="png")
            zf.writestr('thumb.png', thumbnail_img)
            elite = get_elite(mapelites=current_mapelites,
                              bin_idx=_switch([selected_bins[-1]])[0],
                              pop='feasible')
            tmp = CandidateSolution(string=elite.string)
            tmp.ll_string = elite.ll_string
            tmp.base_color = elite.base_color
            current_mapelites.lsystem._set_structure(cs=tmp)
            hullbuilder = HullBuilder(erosion_type=current_mapelites.hull_builder.erosion_type,
                                      apply_erosion=True,
                                      apply_smoothing=True)
            download_semaphore.unlock()
            download_semaphore._running = 'YES'           
            logging.getLogger('webapp').debug(f'[{__name__}.write_archive] {download_semaphore.is_locked=}')
            hullbuilder.add_external_hull(tmp.content)
            tmp.content.set_color(tmp.base_color)
            logging.getLogger('webapp').debug(f'[{__name__}.write_archive] {tmp.string=}; {tmp.content=}; {tmp.base_color=}')
            zf.writestr('bp.sbc', convert_structure_to_xml(structure=tmp.content, name=f'My Spaceship ({rngseed}) (exp{exp_n})'))
            content_properties = {
                'string': tmp.string,
                'base_color': tmp.base_color.as_dict()
            }
            zf.writestr(f'spaceship_{rngseed}_exp{exp_n}', json.dumps(content_properties))
            download_semaphore._running = 'NO'
    
    if selected_bins:
        logging.getLogger('webapp').info(f'Your selected spaceship will be downloaded shortly.')
        return dcc.send_bytes(write_archive, f'MySpaceship_{rngseed}_exp{exp_n}_gen{gen_counter}.zip'), '\n\n'
    else:
        return None, '\n\n'

@app.callback(
    Output("consent-yes", "disabled"),
    Output("consent-no", "disabled"),
    Input("consent-yes", "n_clicks"),
    Input("consent-no", "n_clicks"),
    prevent_initial_call=True
)
def disable_privacy_modal(ny, nn):
    return True, True


@app.callback(Output('step-btn', 'disabled'),
              Output('download-btn', 'disabled'),
              Output('popdownload-btn', 'disabled'),
              Output('rand-step-btn', 'disabled'),
              Output('selection-clr-btn', 'disabled'),
              Output('selection-btn', 'disabled'),
              Output('reset-btn', 'disabled'),
              Output('subdivide-btn', 'disabled'),
              Output('download-mapelites-btn', 'disabled'),
              Output('update-rules-btn', 'disabled'),
              Output('popupload-data', 'disabled'),
              Output('population-dropdown', 'disabled'),
              Output('metric-dropdown', 'disabled'),
              Output('method-radio', 'options'),
              Output('b0-dropdown', 'disabled'),
              Output('b1-dropdown', 'disabled'),
              Output('lsystem-modules', 'options'),
              Output('emitter-dropdown', 'disabled'),
              Output('symmetry-dropdown', 'disabled'),
              Output('symmetry-radio', 'options'),
              Output('color-picker', 'disabled'),
              Output('heatmap-plot-container', 'style'),
              Output({'type': 'fitness-sldr', 'index': ALL}, 'disabled'),

              State({'type': 'fitness-sldr', 'index': ALL}, 'disabled'),
              State('method-radio', 'options'),
              State('lsystem-modules', 'options'),
              State('symmetry-radio', 'options'),
                   
              Input('interval2', 'n_intervals')
              )
def update_btsn_state(fdis, ms, lsysms, symms,
                      ni):
    # non-definitive solution, see: https://github.com/plotly/dash-table/issues/925, https://github.com/plotly/dash/issues/1861
    # long_callback and background callback also do not work (infinite redeployment of webapp)
    global process_semaphore
    
    running_something = process_semaphore.is_locked
    
    for o in ms:
        o['disabled'] = running_something
    for o in symms:
        o['disabled'] = running_something
    for o in lsysms:
        o['disabled'] = running_something
    
    btns = {
        'step-btn.disabled': running_something or (user_study_mode and gen_counter >= N_GENS_ALLOWED),
        'download-btn.disabled': running_something or download_semaphore._running == 'YES',
        'popdownload-btn.disabled': running_something,
        'rand-step-btn.disabled': running_something,
        'selection-clr-btn.disabled': running_something,
        'selection-btn.disabled': running_something,
        'reset-btn.disabled': running_something,
        'subdivide-btn.disabled': running_something,
        'download-mapelites-btn.disabled': running_something,
        'update-rules-btn.disabled': running_something,
        'popupload-data.disabled': running_something,
        'population-dropdown.disabled': running_something,
        'metric-dropdown.disabled': running_something,
        'method-radio.options': ms,
        'b0-dropdown.disabled': running_something,
        'b1-dropdown.disabled': running_something,
        'lsystem-modules.options': lsysms,
        'emitter-dropdown.disabled': running_something,
        'symmetry-dropdown.disabled': running_something,
        'symmetry-radio.options': symms,
        'color-picker.disabled': running_something,
        'heatmap-plot-container.style': {'visibility': 'visible' if running_something else 'hidden',
                                         'display': 'grid' if running_something else 'none',
                                        #  'background': '#ffffff11',  # decomment for debugging purposes
                                         'pointer-events': 'auto',
                                         'z-index': 1 if running_something else -1},
        'fitness-sldr.disabled': [running_something] * len(fdis)
    }
    
    return tuple(btns.values())


def _switch(ls: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    res = []
    for e in ls:
        res.append((e[1], e[0]))
    return res


def _format_bins(mapelites: MAPElites,
                bins_idx_list: List[Tuple[int, int]],
                str_prefix: str,
                do_switch: bool = True,
                filter_out_empty: bool = True) -> Tuple[List[Tuple[int, int]], str]:
    bins_list: List[MAPBin] = [mapelites.bins[j, i] if do_switch else mapelites.bins[i, j] for (i, j) in bins_idx_list]
    sel_bins_str = f'{str_prefix}'
    for b in bins_list:
        i, j = b.bin_idx
        if filter_out_empty:
            if b.non_empty(pop='feasible') or b.non_empty(pop='infeasible'):
                i, j = (j, i) if do_switch else (i, j)
                bc1 = np.sum([mbin.bin_size[0] for mbin in mapelites.bins[:i, j]])
                bc2 = np.sum([mbin.bin_size[1] for mbin in mapelites.bins[i, :j]])
                sel_bins_str += f' {(i, j)} [{bc1}:{bc2}];'
            elif b.bin_idx in bins_idx_list:
                bins_idx_list.remove((i, j))
        else:
            bc1 = np.sum([mbin.bin_size[0] for mbin in mapelites.bins[:i, j]])
            bc2 = np.sum([mbin.bin_size[1] for mbin in mapelites.bins[i, :j]])
            sel_bins_str += f' {(i, j)} [{bc1}:{bc2}];'
    return bins_idx_list, sel_bins_str


def _build_heatmap(mapelites: MAPElites,
                   pop_name: str,
                   metric_name: str,
                   method_name: str) -> go.Figure:
    global gen_counter
    global selected_bins
    
    valid_bins = [x.bin_idx for x in mapelites._valid_bins()]
    metric = hm_callback_props['metric'][metric_name]
    use_mean = hm_callback_props['method'][method_name]
    population = hm_callback_props['pop'][pop_name]
    # build heatmap
    disp_map = np.zeros(shape=mapelites.bins.shape)
    labels = np.zeros(shape=(mapelites.bins.shape[1], mapelites.bins.shape[0], 2))
    text = []
    x_labels = np.cumsum([0] + mapelites.bin_sizes[0][:-1]) + mapelites.b_descs[0].bounds[0]
    y_labels = np.cumsum([0] + mapelites.bin_sizes[1][:-1]) + mapelites.b_descs[1].bounds[0]
    for i in range(mapelites.bins.shape[0]):
        for j in range(mapelites.bins.shape[1]):
            v = mapelites.bins[i, j].get_metric(metric=metric['name'],
                                                use_mean=use_mean,
                                                population=population)
            disp_map[i, j] = v
            s = ''
            if mapelites.bins[i, j].non_empty(pop='feasible'):
                if (i, j) in valid_bins:
                    s = '▣' if gen_counter > 0 and mapelites.bins[i, j].new_elite[population] else s
                    s = '☑' if (j, i) in selected_bins else s                    
            if j == 0:
                text.append([s])
            else:
                text[-1].append(s)
            labels[j, i, 0] = x_labels[i]
            labels[j, i, 1] = y_labels[j]
    # plot
    hovertemplate = f'{mapelites.b_descs[0].name}: X<br>{mapelites.b_descs[1].name}: Y<br>{metric_name}: Z<extra></extra>'
    hovertemplate = hovertemplate.replace('X', '%{customdata[0]}').replace('Y', '%{customdata[1]}').replace('Z', '%{z}')
    heatmap = go.Figure(
        data=go.Heatmap(
            z=disp_map,
            zmin=0,
            zmax=hm_callback_props['metric'][metric_name]['zmax'][population],
            x=np.arange(disp_map.shape[0]),
            y=np.arange(disp_map.shape[1]),
            hoverongaps=False,
            colorscale=hm_callback_props['metric'][metric_name]['colorscale'],
            text=text,
            texttemplate='%{text}',
            textfont={"color": 'rgba(238, 238, 238, 1.)'},
            colorbar={"title": {"text": "Fitness", "side": "right"}, 'orientation': 'v'},
            customdata=labels,
            ))
    heatmap.update_xaxes(title=dict(text=mapelites.b_descs[0].name))
    heatmap.update_yaxes(title=dict(text=mapelites.b_descs[1].name))
    heatmap.update_coloraxes(colorbar_title_text=metric_name)
    heatmap.update_layout(autosize=False,
                          dragmode='pan',
                          clickmode='event+select',
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          template='plotly_dark',
                          margin=go.layout.Margin(
                              l=0,
                              r=0,
                              b=0,
                              t=0))
    heatmap.update_traces(hovertemplate=hovertemplate)
    heatmap.update_traces(selector=dict(type='heatmap'))
    heatmap.update_layout(
        xaxis={
            'tickvals': np.arange(disp_map.shape[0]),
            'ticktext': x_labels
        },
        yaxis={
            'tickvals': np.arange(disp_map.shape[1]),
            'ticktext': y_labels
        },
    )
    
    return heatmap


def _is_base_block(block_type: str) -> bool:
    """Check if the block is a base block. Base blocks are non-functional, structural blocks.

    Args:
        block_type (str): The type of the block.

    Returns:
        bool: Whether the block is a base block.
    """
    return block_type.endswith("Block") or block_type.endswith("Slope") or block_type.endswith("Corner") or block_type.endswith("CornerInv")


def _get_elite_content(mapelites: MAPElites,
                       bin_idx: Optional[Tuple[int, int]],
                       pop: str) -> go.Scatter3d:
    if bin_idx is not None:
        # get elite content
        elite = get_elite(mapelites=mapelites,
                          bin_idx=bin_idx,
                          pop=pop)
        
        structure = elite.content
        content = structure.as_grid_array
        arr = np.nonzero(content)
        x, y, z = arr
        cs = [content[i, j, k] for i, j, k in zip(x, y, z)]
        ss = [structure._clean_label(list(block_definitions.keys())[v - 1]) for v in cs]
        custom_colors = []
        for (i, j, k) in zip(x, y, z):
            b = structure._blocks[(i * structure.grid_size, j * structure.grid_size, k * structure.grid_size)]
            if _is_base_block(block_type=b.block_type):
                custom_colors.append(f'rgb{b.color.as_tuple()}')
            else:
                custom_colors.append(block_to_colour.get(structure._clean_label(b.block_type), '#ff0000'))
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
                    # 'tickmode': 'array',
                    'tickvals': show_x,
                    'ticktext': [structure.grid_size * i for i in show_x],
                },
                yaxis={
                    # 'tickmode': 'array',
                    'tickvals': show_y,
                    'ticktext': [structure.grid_size * i for i in show_y],
                },
                zaxis={
                    # 'tickmode': 'array',
                    'tickvals': show_z,
                    'ticktext': [structure.grid_size * i for i in show_z],
                }
            )
        )
    else:
        fig = go.Figure()
        
        fig.add_scatter3d(x=np.zeros(0, dtype=object),
                          y=np.zeros(0, dtype=object),
                          z=np.zeros(0, dtype=object))
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


def _apply_step(mapelites: MAPElites,
                selected_bins: List[Tuple[int, int]],
                gen_counter: int,
                only_human: bool = False,
                only_emitter: bool = False) -> bool:
    global step_progress
    perc_step = 100 / (1 + N_EMITTER_STEPS)
    
    valid = True
    if mapelites.enforce_qnt:
        valid_bins = [x.bin_idx for x in mapelites._valid_bins()]
        for bin_idx in selected_bins:
            valid &= bin_idx in valid_bins
    if valid:
        logging.getLogger('webapp').info(msg=f'Started step {gen_counter + 1}...')
        # reset bins new_elite flags
        mapelites.update_elites(reset=True)
        step_progress = 0
        if not only_emitter:
            mapelites.interactive_step(bin_idxs=selected_bins,
                                       gen=gen_counter)
        
        step_progress += perc_step
        logging.getLogger('webapp').info(msg=f'Completed step {gen_counter + 1} (created {mapelites.n_new_solutions} solutions); running {N_EMITTER_STEPS} additional emitter steps if available...')
        mapelites.n_new_solutions = 0
        with trange(N_EMITTER_STEPS, desc='Emitter steps: ') as iterations:
            for _ in iterations:
                if not only_human:
                    mapelites.emitter_step(gen=gen_counter)
                step_progress += perc_step
        logging.getLogger('webapp').info(msg=f'Emitter step(s) completed (created {mapelites.n_new_solutions} solutions).')
        mapelites.n_new_solutions = 0
        step_progress = -1
        mapelites.update_elites()
        return True
    else:
        logging.getLogger('webapp').info(msg='Step not applied: invalid bin(s) selected.')
        return False


def __apply_step(**kwargs) -> Dict[str, Any]:
    global consent_ok
    global current_mapelites
    global gdev_mode
    global gen_counter
    global n_spaceships_inspected
    global selected_bins
    global time_elapsed
    global user_study_mode
    
    cs_properties = kwargs['cs_properties']
    cs_string = kwargs['cs_string']
    curr_content = kwargs['curr_content']
    curr_heatmap = kwargs['curr_heatmap']
    eoe_modal_show = kwargs['eoe_modal_show']
    nbs_err_modal_show = kwargs['nbs_err_modal_show']
    dlbtn_label = kwargs['dlbtn_label']
    
    if selected_bins or kwargs['event_trig'] == 'rand-step-btn':
        s = time.perf_counter()
        res = _apply_step(mapelites=current_mapelites,
                          selected_bins=_switch(selected_bins),
                          gen_counter=gen_counter,
                          only_human=kwargs['event_trig'] == 'step-btn' and not user_study_mode and not gdev_mode,
                          only_emitter=kwargs['event_trig'] == 'rand-step-btn' and not user_study_mode and not gdev_mode)
        if res:
            elapsed = time.perf_counter() - s
            gen_counter += 1
            # update metrics if user consented to privacy
            if consent_ok:
                # n_spaceships_inspected.add(1)
                time_elapsed.add(elapsed)
            if selected_bins:
                rem_idxs = []
                for i, b in enumerate(selected_bins):
                    # remove preview and properties if last selected bin is now invalid
                    lb = _switch([b])[0]
                    if lb not in [b.bin_idx for b in current_mapelites._valid_bins()]:
                        rem_idxs.append(i)
                for i in reversed(rem_idxs):
                    selected_bins.pop(i)
                if selected_bins == []:
                        curr_content = _get_elite_content(mapelites=current_mapelites,
                                                        bin_idx=None,
                                                        pop='')
                        cs_string = ''
                        cs_properties = get_properties_table()
                else:
                    lb = _switch([selected_bins[-1]])[0]
                    if current_mapelites.bins[selected_bins[-1]].new_elite[hm_callback_props['pop'][kwargs['pop_name']]]:
                        curr_content = _get_elite_content(mapelites=current_mapelites,
                                                        bin_idx=lb,
                                                        pop='feasible' if kwargs['pop_name'] == 'Feasible' else 'infeasible')
                        elite = get_elite(mapelites=current_mapelites,
                                        bin_idx=lb,
                                        pop='feasible' if kwargs['pop_name'] == 'Feasible' else 'infeasible')
                        cs_string = elite.string
                        cs_properties = get_properties_table(cs=elite)
            # prompt user to download content if reached end of generations
            if user_study_mode and gen_counter == N_GENS_ALLOWED:
                eoe_modal_show = True
            # update heatmap
            curr_heatmap = _build_heatmap(mapelites=current_mapelites,
                                          pop_name=kwargs['pop_name'],
                                          metric_name=kwargs['metric_name'],
                                          method_name=kwargs['method_name'])
            logging.getLogger('webapp').debug(msg=f'[{__name__}.__apply_step] {elapsed=}; {gen_counter=}; {selected_bins=}')
    else:
        logging.getLogger('webapp').error(msg=f'Step not applied: no bin(s) selected.')
        nbs_err_modal_show = True
    
    if user_study_mode and gen_counter == N_GENS_ALLOWED:
        if exp_n + 1 == len(my_emitterslist):
            dlbtn_label = 'Download and Switch to User Mode'
        else:
            dlbtn_label = 'Download and Start Next Experiment'
    
    return {
        'content-plot.figure': curr_content,
        'content-string.value': cs_string,
        'eoe-modal.is_open': eoe_modal_show,
        'heatmap-plot.figure': curr_heatmap,
        'nbs-err-modal.is_open': nbs_err_modal_show,
        'spaceship-properties.children': cs_properties,
        'download-btn.children': dlbtn_label
    }


def __reset(**kwargs) -> Dict[str, Any]:
    global consent_ok
    global current_mapelites
    global gen_counter
    global n_spaceships_inspected
    global time_elapsed
    global selected_bins
    
    logging.getLogger('webapp').info(msg='Started resetting all bins (this may take a while)...')
    current_mapelites.reset()
    current_mapelites.hull_builder.apply_smoothing = False
    logging.getLogger('webapp').info(msg='Reset completed.')
    gen_counter = 0
    selected_bins = []
    if consent_ok:
        n_spaceships_inspected.reset()
        time_elapsed.reset()

    return {
        'heatmap-plot.figure': _build_heatmap(mapelites=current_mapelites,
                                              pop_name=kwargs['pop_name'],
                                              metric_name=kwargs['metric_name'],
                                              method_name=kwargs['method_name'])
    }


def __bc_change(**kwargs) -> Dict[str, Any]:
    global current_mapelites
    
    event_trig = kwargs['event_trig']
    b0 = kwargs['b0']
    b1 = kwargs['b1']
    curr_heatmap = kwargs['curr_heatmap']
        
    if event_trig.startswith('bc0') or event_trig.startswith('bc1'):
        if event_trig.startswith('bc0'):
            b0 = event_trig.replace('bc0-', '').replace('_', ' / ').replace('-', ' ')
        else:
            b1 = event_trig.replace('bc1-', '').replace('_', ' / ').replace('-', ' ')
        logging.getLogger('webapp').info(msg=f'Updating feature descriptors to ({b0}, {b1})...')
        b0 = behavior_descriptors[[b.name for b in behavior_descriptors].index(b0)]
        b1 = behavior_descriptors[[b.name for b in behavior_descriptors].index(b1)]
        current_mapelites.update_behavior_descriptors((b0, b1))
        curr_heatmap = _build_heatmap(mapelites=current_mapelites,
                                      pop_name=kwargs['pop_name'],
                                      metric_name=kwargs['metric_name'],
                                      method_name=kwargs['method_name'])
        logging.getLogger('webapp').info(msg='Feature descriptors update completed.')
    else:
        logging.getLogger('webapp').error(msg=f'Could not change BC: passed unrecognized value ({event_trig}).')
    
    return {
        'heatmap-plot.figure': curr_heatmap,
        }


def __subdivide(**kwargs) -> Dict[str, Any]:
    global current_mapelites
    global selected_bins
    
    curr_heatmap = kwargs['curr_heatmap']
    
    bin_idxs = [(x[1], x[0]) for x in selected_bins]
    for bin_idx in bin_idxs:
        logging.getLogger('webapp').debug(msg=f'[{__name__}.__subdivide] Subdividing {bin_idx=}')
        current_mapelites.subdivide_range(bin_idx=bin_idx)
    curr_heatmap = _build_heatmap(mapelites=current_mapelites,
                                    pop_name=kwargs['pop_name'],
                                    metric_name=kwargs['metric_name'],
                                    method_name=kwargs['method_name'])
    logging.getLogger('webapp').info(msg=f'Subdivided bin(s): {selected_bins}.')
    selected_bins = []
    
    return {
        'heatmap-plot.figure': curr_heatmap,
        }


def __lsystem_modules(**kwargs) -> Dict[str, Any]:
    global current_mapelites
    
    modules = kwargs['modules']
    
    all_modules = [x for x in current_mapelites.lsystem.modules]
    names = [x.name for x in all_modules]
    for i, module in enumerate(names):
        if module in modules and not all_modules[i].active:
            # activate module
            current_mapelites.toggle_module_mutability(module=module)
            logging.getLogger('webapp').debug(msg=f'[{__name__}.__subdivide] Enabled {module}')
            break
        elif module not in modules and all_modules[i].active:
            # deactivate module
            current_mapelites.toggle_module_mutability(module=module)
            logging.getLogger('webapp').debug(msg=f'[{__name__}.__subdivide] Disabled {module}')
            break
    logging.getLogger('webapp').info(msg=f'L-system modules updated')

    return {}


def __update_rules(**kwargs) -> Dict[str, Any]:
    global current_mapelites
    
    rules = kwargs['rules']
    
    new_rules = StochasticRules()
    for rule in rules.split('\n'):
        lhs, p, rhs = rule.strip().split(' ')
        new_rules.add_rule(lhs=lhs,
                        rhs=rhs,
                        p=float(p))
    try:
        new_rules.validate()
        current_mapelites.lsystem.hl_solver.parser.rules = new_rules
        logging.getLogger('webapp').info(msg=f'L-system rules updated.')
        return True
    except AssertionError as e:
        logging.getLogger('webapp').info(msg=f'Failed updating L-system rules ({e}).')
    
    return {
        'hl-rules.value': str(current_mapelites.lsystem.hl_solver.parser.rules)
    }


def __fitness_weights(**kwargs) -> Dict[str, Any]:
    global current_mapelites

    curr_heatmap = kwargs['curr_heatmap']
    weights = kwargs['weights']
    
    current_mapelites.update_fitness_weights(weights=weights)
    logging.getLogger('webapp').info(msg='Updated fitness functions weights.')
    hm_callback_props['metric']['Fitness']['zmax']['feasible'] = sum([x.weight * x.bounds[1] for x in current_mapelites.feasible_fitnesses]) + current_mapelites.nsc
    
    curr_heatmap = _build_heatmap(mapelites=current_mapelites,
                                    pop_name=kwargs['pop_name'],
                                    metric_name=kwargs['metric_name'],
                                    method_name=kwargs['method_name'])
    
    return {
        'heatmap-plot.figure': curr_heatmap,
        }


def __update_heatmap(**kwargs) -> Dict[str, Any]:
    global current_mapelites
    
    event_trig = kwargs['event_trig']
    pop_name = kwargs['pop_name']
    metric_name = kwargs['metric_name']
    method_name = kwargs['method_name']
    
    if event_trig == 'population-feasible':
        pop_name = 'Feasible'
    elif event_trig == 'population-infeasible':
        pop_name = 'Infeasible'
    elif event_trig == 'metric-fitness':
        metric_name = 'Fitness'
    elif event_trig == 'metric-age':
        metric_name = 'Age'
    elif event_trig == 'metric-coverage':
        metric_name = 'Coverage'
    logging.getLogger('webapp').debug(msg=f'[{__name__}.__update_heatmap] {pop_name=}; {metric_name=}; {method_name=}')

    curr_heatmap = _build_heatmap(mapelites=current_mapelites,
                                    pop_name=pop_name,
                                    metric_name=metric_name,
                                    method_name=method_name)
    return {
        'heatmap-plot.figure': curr_heatmap,
        }
    
    
def __apply_symmetry(**kwargs) -> Dict[str, Any]:
    global current_mapelites
    global selected_bins
    
    event_trig = kwargs['event_trig']
    symm_orientation = kwargs['symm_orientation']
    
    logging.getLogger('webapp').info(msg=f'Updating all solutions to enforce symmetry...')
    if event_trig == 'symmetry-none':
        symm_axis = 'None'
    elif event_trig == 'symmetry-x':
        symm_axis = 'X-axis'
    elif event_trig == 'symmetry-y':
        symm_axis = 'Y-axis'
    elif event_trig == 'symmetry-z':
        symm_axis = 'Z-axis'
    logging.getLogger('webapp').debug(msg=f'[{__name__}.__apply_symmetry] {symm_axis=}; {symm_orientation=}')
        
    current_mapelites.reassign_all_content(sym_axis=symm_axis[0].lower() if symm_axis != "None" else None,
                                           sym_upper=symm_orientation == 'Upper')
    curr_content = _get_elite_content(mapelites=current_mapelites,
                                      bin_idx=None,
                                      pop=None)
    logging.getLogger('webapp').info(msg=f'Symmetry enforcement completed.')
    
    selected_bins = []
    
    return {
        'content-plot.figure': curr_content,
        'content-string.value': '',
        'spaceship-properties.children': get_properties_table(),
        'symmetry-dropdown.label': symm_axis
        }


def __update_content(**kwargs) -> Dict[str, Any]:
    global current_mapelites
    global selected_bins
    
    curr_heatmap = kwargs['curr_heatmap']
    curr_content = kwargs['curr_content']
    cs_string = kwargs['cs_string']
    cs_properties = kwargs['cs_properties']
        
    i, j = kwargs['clickData']['points'][0]['x'], kwargs['clickData']['points'][0]['y']
    if current_mapelites.bins[j, i].non_empty(pop='feasible' if kwargs['pop_name'] == 'Feasible' else 'infeasible'):
        if (j, i) in [b.bin_idx for b in current_mapelites._valid_bins()]:
            curr_content = _get_elite_content(mapelites=current_mapelites,
                                              bin_idx=(j, i),
                                              pop='feasible' if kwargs['pop_name'] == 'Feasible' else 'infeasible')
            if consent_ok:
                n_spaceships_inspected.add(1)
            if not current_mapelites.enforce_qnt and selected_bins != []:
                if (i, j) not in selected_bins:
                    selected_bins.append((i, j))
                else:
                    selected_bins.remove((i, j))
            else:
                selected_bins = [(i, j)]
            cs_string = ''
            cs_properties = get_properties_table()
            if len(selected_bins) > 0:
                elite = get_elite(mapelites=current_mapelites,
                                  bin_idx=_switch([selected_bins[-1]])[0],
                                  pop='feasible' if kwargs['pop_name'] == 'Feasible' else 'infeasible')
                cs_string = elite.string
                cs_properties = get_properties_table(cs=elite)
                curr_heatmap = _build_heatmap(mapelites=current_mapelites,
                                              pop_name=kwargs['pop_name'],
                                              metric_name=kwargs['metric_name'],
                                              method_name=kwargs['method_name'])
    else:
        logging.getLogger('webapp').error(msg=f'[{__name__}.__update_content] Empty bin selected: {(i, j)=}')
    
    return {
        'heatmap-plot.figure': curr_heatmap,
        'content-plot.figure': curr_content,
        'content-string.value': cs_string,
        'spaceship-properties.children': cs_properties,
        }


def __selection(**kwargs) -> Dict[str, Any]:
    global current_mapelites
    global selected_bins
    
    current_mapelites.enforce_qnt = not current_mapelites.enforce_qnt
    logging.getLogger('webapp').info(msg=f'MAP-Elites single bin selection set to {current_mapelites.enforce_qnt}.')
    if current_mapelites.enforce_qnt and selected_bins:
        selected_bins = [selected_bins[-1]]
    
    return {}


def __clear_selection(**kwargs) -> Dict[str, Any]:
    global current_mapelites
    global selected_bins
    
    logging.getLogger('webapp').info(msg='Cleared bins selection.')
    selected_bins = []
    
    return {
        'content-plot.figure':  _get_elite_content(mapelites=current_mapelites,
                                                   bin_idx=None,
                                                   pop=None),
        'content-string.value': '',
        'spaceship-properties.children': get_properties_table(),
        }


def __emitter(**kwargs) -> Dict[str, Any]:
    global current_mapelites
    global selected_bins

    emitter_name = kwargs['emitter_name']
    
    if emitter_name == 'Random':
        current_mapelites.emitter = RandomEmitter()
        logging.getLogger('webapp').info(msg=f'Emitter set to {emitter_name}')
    if emitter_name == 'Greedy':
        current_mapelites.emitter = GreedyEmitter()
        logging.getLogger('webapp').info(msg=f'Emitter set to {emitter_name}')
    elif emitter_name == 'Preference-matrix':
        current_mapelites.emitter = HumanPrefMatrixEmitter()
        current_mapelites.emitter._build_pref_matrix(bins=current_mapelites.bins)
        logging.getLogger('webapp').info(msg=f'Emitter set to {emitter_name}')
    elif emitter_name == 'Contextual Bandit':
        current_mapelites.emitter = ContextualBanditEmitter()
        logging.getLogger('webapp').info(msg=f'Emitter set to {emitter_name}')
    elif emitter_name == 'Preference Bandit':
        current_mapelites.emitter = PreferenceBanditEmitter()
        logging.getLogger('webapp').info(msg=f'Emitter set to {emitter_name}')
    elif emitter_name == 'None':
        current_mapelites.emitter = HumanEmitter()
        logging.getLogger('webapp').info(msg=f'Emitter set to {emitter_name}')
    else:
        logging.getLogger('webapp').error(msg=f'[{__name__}.__emitter] Unrecognized {emitter_name=}')

    return {}


def __content_download(**kwargs) -> Dict[str, Any]:
    global current_mapelites
    global selected_bins
    global user_study_mode
    global gen_counter
    global exp_n
    global my_emitterslist
    global consent_ok
    global time_elapsed
    global n_spaceships_inspected
    global rngseed
    global download_semaphore

    cs_string = kwargs['cs_string']
    cs_properties = kwargs['cs_properties']
    curr_heatmap = kwargs['curr_heatmap']
    curr_content = kwargs['curr_content']
    eous_modal_show = kwargs['eous_modal_show']
    qs_um_modal_show = kwargs['qs_um_modal_show']
    nbs_err_modal_show = kwargs['nbs_err_modal_show']
    rand_step_btn_style = kwargs['rand_step_btn_style']
    reset_btn_style = kwargs['reset_btn_style']
    exp_progress_style = kwargs['exp_progress_style']
    metrics_dl = None
    dlbtn_label = kwargs['dlbtn_label']
    
    # if cs_string != '':
    if selected_bins:
        
        logging.getLogger('webapp').debug(f'[{__name__}.__content_download] (pre-check) {download_semaphore.is_locked=}')
        while download_semaphore.is_locked:
            pass
        logging.getLogger('webapp').debug(f'[{__name__}.__content_download] (post-check) {download_semaphore.is_locked=}')
        download_semaphore.lock(name=download_semaphore._running)
        logging.getLogger('webapp').debug(f'[{__name__}.__content_download] (re-lock) {download_semaphore.is_locked=}')
        
        if user_study_mode and gen_counter == N_GENS_ALLOWED:
            # time.sleep(2)
            exp_n += 1
            # check end of user study
            if exp_n >= len(my_emitterslist):
                curr_heatmap = go.Figure(
                    data=go.Heatmap(
                        z=np.zeros(0, dtype=object),
                        x=np.zeros(0, dtype=object),
                        y=np.zeros(0, dtype=object),
                        hoverongaps=False,
                        ))
                curr_content = _get_elite_content(mapelites=current_mapelites,
                                                  bin_idx=None,
                                                  pop=None)
                cs_string = ''
                cs_properties = get_properties_table()
                if consent_ok:
                    metrics_dl = dict(content=json.dumps({
                        'time_elapsed': time_elapsed.get_averages(),
                        'n_interactions': n_spaceships_inspected.get_averages()
                        }),
                                      filename=f'user_metrics_{rngseed}')
                else:
                    metrics_dl = None
                logging.getLogger('webapp').info(f'Reached end of all experiments! Please go back to the questionnaire to continue the evaluation.')
                eous_modal_show = True
                qs_um_modal_show = True
                user_study_mode = False
                dlbtn_label = 'Download'
                selected_bins = []
                logging.getLogger('webapp').info(msg='Initializing a new population; this may take a while...')
                current_mapelites.reset()
                current_mapelites.hull_builder.apply_smoothing = False
                current_mapelites.emitter = RandomEmitter()
                rand_step_btn_style, reset_btn_style, exp_progress_style = {}, {}, hidden_style
                curr_heatmap = _build_heatmap(mapelites=current_mapelites,
                                              pop_name=kwargs['pop_name'],
                                              metric_name=kwargs['metric_name'],
                                              method_name=kwargs['method_name'])
                curr_content = _get_elite_content(mapelites=current_mapelites,
                                                  bin_idx=None,
                                                  pop=None)
                logging.getLogger('webapp').info(msg='Initialization completed.')
            else:
                logging.getLogger('webapp').info(msg=f'Reached end of experiment {exp_n}! Loading the next experiment...')
                gen_counter = 0
                dlbtn_label = 'Download'
                selected_bins = []
                if consent_ok:
                    logging.getLogger('webapp').info(msg='Loading next population...')
                    current_mapelites.reset(lcs=[])
                    current_mapelites.hull_builder.apply_smoothing = False
                    current_mapelites.load_population(filename=my_emitterslist[exp_n])
                    logging.getLogger('webapp').info(msg='Next population loaded.')
                    n_spaceships_inspected.new_generation()
                    time_elapsed.new_generation()
                    logging.getLogger('webapp').info(msg='Next experiment loaded. Please fill out the questionnaire before continuing.')
                else:
                    logging.getLogger('webapp').info(msg='Initializing a new population; this may take a while...')
                    current_mapelites.reset()
                    current_mapelites.hull_builder.apply_smoothing = False
                    logging.getLogger('webapp').info(msg='Initialization completed.')
                curr_heatmap = _build_heatmap(mapelites=current_mapelites,
                                              pop_name=kwargs['pop_name'],
                                              metric_name=kwargs['metric_name'],
                                              method_name=kwargs['method_name'])
                curr_content = _get_elite_content(mapelites=current_mapelites,
                                                  bin_idx=None,
                                                  pop=None)
                cs_string = ''
                cs_properties = get_properties_table()
            # update base color on new experiment switch
            _update_base_color(color=base_color)
    else:
        nbs_err_modal_show = True

    logging.getLogger('webapp').debug(f'[{__name__}.__content_download] {selected_bins=}; {exp_n=}; {gen_counter=}')
    
    return {
        'heatmap-plot.figure': curr_heatmap,
        'content-plot.figure': curr_content,
        'content-string.value': cs_string,
        'spaceship-properties.children': cs_properties,
        'nbs-err-modal.is_open': nbs_err_modal_show,
        'eous-modal.is_open': eous_modal_show,
        'quickstart-usermode-modal.is_open': qs_um_modal_show,
        'rand-step-btn-div.style': rand_step_btn_style,
        'reset-btn-div.style': reset_btn_style,
        'exp-progress-div.style': exp_progress_style,
        'download-metrics.data': metrics_dl,
        'download-btn.children': dlbtn_label
    }


def __population_download(**kwargs) -> Dict[str, Any]:
    global current_mapelites
    global rngseed
    
    content_dl = dict(content=json.dumps([b.to_json() for b in current_mapelites.bins.flatten().tolist()]),
                      filename=f'population_{rngseed}_exp{exp_n}_{current_mapelites.emitter.name}.json')
    logging.getLogger('webapp').info(f'The population will be downloaded shortly.')
    return {
        'download-population.data': content_dl
    }


def __population_upload(**kwargs) -> Dict[str, Any]:
    global current_mapelites
    
    upload_contents = kwargs['upload_contents']
    
    _, upload_contents = upload_contents.split(',')
    upload_contents = base64.b64decode(upload_contents).decode()        
    all_bins = np.asarray([MAPBin.from_json(x) for x in json.loads(upload_contents)])
    current_mapelites.reset(lcs=[])
    all_bins = all_bins.reshape(current_mapelites.bin_qnt)
    current_mapelites.bins = all_bins
    current_mapelites.reassign_all_content()
    logging.getLogger('webapp').info(msg=f'Set population from file successfully.')

    return {
        'heatmap-plot.figure': _build_heatmap(mapelites=current_mapelites,
                                pop_name=kwargs['pop_name'],
                                metric_name=kwargs['metric_name'],
                                method_name=kwargs['method_name'])
        }


def __consent(**kwargs) -> Dict[str, Any]:
    global current_mapelites
    global consent_ok
    global gen_counter
    global user_study_mode
    
    nclicks_yes = kwargs['nclicks_yes']
    nclicks_no = kwargs['nclicks_no']
    qs_modal_show = kwargs['qs_modal_show']
    qs_um_modal_show = kwargs['qs_um_modal_show']
    rand_step_btn_style = kwargs['rand_step_btn_style']
    reset_btn_style = kwargs['reset_btn_style']
    exp_progress_style = kwargs['exp_progress_style']
    cm_modal_show = kwargs['cm_modal_show']
    study_style = kwargs['study_style']
    
    consent_ok = True if nclicks_yes else False if nclicks_no else None
    if nclicks_yes:
        logging.getLogger('webapp').info(msg=f'Thank you for participating in the user study! Please do not refresh the page.')
        logging.getLogger('webapp').info(msg='Loading population...')
        current_mapelites.reset(lcs=[])
        current_mapelites.hull_builder.apply_smoothing = False
        current_mapelites.load_population(filename=my_emitterslist[exp_n])
        logging.getLogger('webapp').info(msg='Population loaded.')
        qs_modal_show = True
    else:
        logging.getLogger('webapp').info(msg=f'No user data will be collected during this session. Please do not refresh the page.')
        logging.getLogger('webapp').info(msg='Initializing population; this may take a while...')
        current_mapelites.emitter = RandomEmitter()
        current_mapelites.reset()
        current_mapelites.hull_builder.apply_smoothing = False
        logging.getLogger('webapp').info(msg='Initialization completed.')
        user_study_mode = False
        qs_um_modal_show = True
        rand_step_btn_style, reset_btn_style, exp_progress_style, study_style = {}, {}, hidden_style, hidden_style
    cm_modal_show = False
    gen_counter = 0
    
    return {
        'heatmap-plot.figure': _build_heatmap(mapelites=current_mapelites,
                                              pop_name=kwargs['pop_name'],
                                              metric_name=kwargs['metric_name'],
                                              method_name=kwargs['method_name']),
        'consent-modal.is_open': cm_modal_show,
        'quickstart-modal.is_open': qs_modal_show,
        'quickstart-usermode-modal.is_open': qs_um_modal_show,
        'rand-step-btn-div.style': rand_step_btn_style,
        'reset-btn-div.style': reset_btn_style,
        'exp-progress-div.style': exp_progress_style,
        'study-progress-div.style': study_style
        }


def __close_error(**kwargs) -> Dict[str, Any]:
    return {'nbs-err-modal.is_open': False}


def _update_base_color(color: Vec) -> None:
    global current_mapelites
    logging.getLogger('webapp').debug(f'[{__name__}._update_base_color] {color=}')
    for (_, _), b in np.ndenumerate(current_mapelites.bins):
            for cs in [*b._feasible, *b._infeasible]:
                cs.base_color = color
                cs.content.set_color(color)


def __color(**kwargs) -> Dict[str, Any]:
    global current_mapelites
    global base_color
    
    color = kwargs['color']
    curr_content = kwargs['curr_content']
    
    r, g, b = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    new_color = Vec.v3f(r, g, b).scale(1 / 256)
    base_color = new_color
    logging.getLogger('webapp').debug(msg=f'[{__name__}.__color] {base_color=}')
    _update_base_color(color=base_color)
    if selected_bins:
        curr_content =  _get_elite_content(mapelites=current_mapelites,
                                           bin_idx=_switch([selected_bins[-1]])[0],
                                           pop='feasible' if kwargs['pop_name'] == 'Feasible' else 'infeasible')
    return {
        'content-plot.figure': curr_content
    }


def __default(**kwargs) -> Dict[str, Any]:
    global current_mapelites
    
    return {
        'heatmap-plot.figure': _build_heatmap(mapelites=current_mapelites,
                                              pop_name=kwargs['pop_name'],
                                              metric_name=kwargs['metric_name'],
                                              method_name=kwargs['method_name']),
        'content-plot.figure': _get_elite_content(mapelites=current_mapelites,
                                                  bin_idx=None,
                                                  pop=None)
        }


triggers_map = {
    'step-btn': __apply_step,
    'rand-step-btn': __apply_step,
    'reset-btn': __reset,
    'bc0-Major-axis_Medium-axis': __bc_change,
    'bc0-Major-axis_Smallest-axis': __bc_change,
    'bc0-Average-Proportions': __bc_change,
    'bc0-Symmetry': __bc_change,
    'bc1-Major-axis_Medium-axis': __bc_change,
    'bc1-Major-axis_Smallest-axis': __bc_change,
    'bc1-Average-Proportions': __bc_change,
    'bc1-Symmetry': __bc_change,
    'subdivide-btn': __subdivide,
    'lsystem-modules': __lsystem_modules,
    'update-rules-btn': __update_rules,
    'population-feasible': __update_heatmap,
    'population-infeasible': __update_heatmap,
    'metric-fitness': __update_heatmap,
    'metric-age': __update_heatmap,
    'metric-coverage': __update_heatmap,
    'method-radio': __update_heatmap,
    'symmetry-none': __apply_symmetry,
    'symmetry-x': __apply_symmetry,
    'symmetry-y': __apply_symmetry,
    'symmetry-z': __apply_symmetry,
    'symmetry-radio': __apply_symmetry,
    'heatmap-plot': __update_content,
    'population_dropdown': __update_content,
    'selection-btn': __selection,
    'selection-clr-btn': __clear_selection,
    'emitter-dropdown': __emitter,
    'download-btn': __content_download,
    'popdownload-btn': __population_download,
    'popupload-data': __population_upload,
    'consent-yes': __consent,
    'consent-no': __consent,
    'nbs-err-btn': __close_error,
    'color-picker': __color,
    'fitness-sldr': __fitness_weights,
    None: __default
}


@app.callback(Output('heatmap-plot', 'figure'),
              Output('content-plot', 'figure'),
              Output('valid-bins', 'children'),
              Output('hl-rules', 'value'),
              Output('selected-bin', 'children'),
              Output('content-string', 'value'),
              Output('spaceship-properties', 'children'),
              Output('step-spinner', 'children'),
              Output("download-population", "data"),
              Output("download-metrics", "data"),
              Output('population-dropdown', 'label'),
              Output('metric-dropdown', 'label'),
              Output('b0-dropdown', 'label'),
              Output('b1-dropdown', 'label'),
              Output('symmetry-dropdown', 'label'),
              Output("quickstart-modal", "is_open"),
              Output("quickstart-usermode-modal", "is_open"),
              Output("consent-modal", "is_open"),
              Output("nbs-err-modal", "is_open"),
              Output("eoe-modal", "is_open"),
              Output("eous-modal", "is_open"),
              Output("rand-step-btn-div", "style"),
              Output("reset-btn-div", "style"),
              Output("exp-progress-div", "style"),
              Output('study-progress-div', 'style'),
              Output('download-btn', 'children'),
              
              State('heatmap-plot', 'figure'),
              State('hl-rules', 'value'),
              State('content-plot', 'figure'),
              State('content-string', 'value'),
              State('spaceship-properties', 'children'),
              State('population-dropdown', 'label'),
              State('metric-dropdown', 'label'),
              State('b0-dropdown', 'label'),
              State('b1-dropdown', 'label'),
              State('symmetry-dropdown', 'label'),
              State("quickstart-modal", "is_open"),
              State("quickstart-usermode-modal", "is_open"),
              State("consent-modal", "is_open"),
              State("nbs-err-modal", "is_open"),
              State("eoe-modal", "is_open"),
              State("eous-modal", "is_open"),
              State("rand-step-btn-div", "style"),
              State("reset-btn-div", "style"),
              State("exp-progress-div", "style"),
              State('study-progress-div', 'style'),
              State('download-btn', 'children'),
              
              Input('population-feasible', 'n_clicks'),
              Input('population-infeasible', 'n_clicks'),
              Input('metric-fitness', 'n_clicks'),
              Input('metric-age', 'n_clicks'),
              Input('metric-coverage', 'n_clicks'),
              Input('method-radio', 'value'),
              Input('step-btn', 'n_clicks'),
              Input('rand-step-btn', 'n_clicks'),
              Input('reset-btn', 'n_clicks'),
              Input('subdivide-btn', 'n_clicks'),
              Input({'type': 'fitness-sldr', 'index': ALL}, 'value'),
              Input('bc0-Major-axis_Medium-axis', 'n_clicks'),
              Input('bc0-Major-axis_Smallest-axis', 'n_clicks'),
              Input('bc0-Average-Proportions', 'n_clicks'),
              Input('bc0-Symmetry', 'n_clicks'),
              Input('bc1-Major-axis_Medium-axis', 'n_clicks'),
              Input('bc1-Major-axis_Smallest-axis', 'n_clicks'),
              Input('bc1-Average-Proportions', 'n_clicks'),
              Input('bc1-Symmetry', 'n_clicks'),
              Input('lsystem-modules', 'value'),
              Input('update-rules-btn', 'n_clicks'),
              Input('heatmap-plot', 'clickData'),
              Input('selection-btn', 'n_clicks'),
              Input('selection-clr-btn', 'n_clicks'),
              Input('emitter-dropdown', 'label'),
              Input("download-btn", "n_clicks"),
              Input('popdownload-btn', 'n_clicks'),
              Input('popupload-data', 'contents'),
              Input('symmetry-none', 'n_clicks'),
              Input('symmetry-x', 'n_clicks'),
              Input('symmetry-y', 'n_clicks'),
              Input('symmetry-z', 'n_clicks'),
              Input('symmetry-radio', 'value'),
              Input("consent-yes", "n_clicks"),
              Input("consent-no", "n_clicks"),
              Input("nbs-err-btn", "n_clicks"),
              Input('color-picker', 'value'),
              )
def general_callback(curr_heatmap, rules, curr_content, cs_string, cs_properties, pop_name, metric_name, b0, b1, symm_axis, qs_modal_show, qs_um_modal_show, cm_modal_show, nbs_err_modal_show, eoe_modal_show, eous_modal_show, rand_step_btn_style, reset_btn_style, exp_progress_style, study_style, dlbtn_label,
                     pop_feas, pop_infeas, metric_fitness, metric_age, metric_coverage, method_name, n_clicks_step, n_clicks_rand_step, n_clicks_reset, n_clicks_sub, weights, b0_mame, b0_mami, b0_avgp, b0_sym, b1_mame, b1_mami, b1_avgp, b1_sym, modules, n_clicks_rules, clickData, selection_btn, clear_btn, emitter_name, n_clicks_cs_download, n_clicks_popdownload, upload_contents, symm_none, symm_x, symm_y, symm_z, symm_orientation, nclicks_yes, nclicks_no, nbs_btn, color):
    global current_mapelites
    global selected_bins
    
    ctx = dash.callback_context

    if not ctx.triggered:
        event_trig = None
    else:
        event_trig = ctx.triggered[0]['prop_id'].split('.')[0]

    if event_trig not in triggers_map:
        try:
            import ast
            event_trig = ast.literal_eval(event_trig)
            event_trig = event_trig['type']
        except ValueError:
            logging.getLogger('webapp').error(msg=f'[{__name__}.general_callback] Unrecognized {event_trig=}. No operations have been applied!')
    
    vars = locals()
    
    output = {
        'heatmap-plot.figure': curr_heatmap,
        'content-plot.figure': curr_content,
        'valid-bins.children': '',
        'hl-rules.value': rules,
        'selected-bin.children': '',
        'content-string.value': cs_string,
        'spaceship-properties.children': cs_properties,
        'step-spinner.children': '',
        'download-population.data': None,
        'download-metrics.data': None,
        'population-dropdown.label': pop_name,
        'metric-dropdown.label': metric_name,
        'b0-dropdown.label': b0,
        'b1-dropdown.label': b1,
        'symmetry-dropdown.label': symm_axis,
        'quickstart-modal.is_open': qs_modal_show,
        'quickstart-usermode-modal.is_open': qs_um_modal_show,
        'consent-modal.is_open': cm_modal_show if consent_ok is not None else True,
        'nbs-err-modal.is_open': nbs_err_modal_show,
        'eoe-modal.is_open': eoe_modal_show,
        'eous-modal.is_open': eous_modal_show,
        'rand-step-btn-div.style': rand_step_btn_style,
        'reset-btn-div.style': reset_btn_style,
        'exp-progress-div.style': exp_progress_style,
        'study-progress-div.style': study_style,
        'download-btn.children': dlbtn_label,
    }
    
    logging.getLogger('webapp').debug(f'[{__name__}.general_callback] {event_trig=}; {exp_n=}; {gen_counter=}; {selected_bins=}; {process_semaphore.is_locked=}')
    
    if not process_semaphore.is_locked:
        process_semaphore.lock(name=event_trig)

        u = triggers_map[event_trig](**vars)
        for k in u.keys():
            output[k] = u[k]

        selected_bins, selected_bins_str = _format_bins(mapelites=current_mapelites,
                                                        bins_idx_list=selected_bins,
                                                        do_switch=True,
                                                        str_prefix='Selected bin(s):',
                                                        filter_out_empty=True) 
        _, valid_bins_str = _format_bins(mapelites=current_mapelites,
                                         do_switch=False,
                                         bins_idx_list=_switch([x.bin_idx for x in current_mapelites._valid_bins()]),
                                         str_prefix='Valid bins are:',
                                         filter_out_empty=False)
        
        output['selected-bin.children'] = selected_bins_str
        output['valid-bins.children'] = valid_bins_str
        
        process_semaphore.unlock()
    
    return tuple(output.values())
