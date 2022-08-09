import base64
import json
import logging
import os
import random
from re import S
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from zipfile import ZipFile
import time

import dash
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from pcgsepy.common.jsonifier import json_dumps, json_loads
from pcgsepy.config import (BIN_POP_SIZE, CS_MAX_AGE, N_EMITTER_STEPS,
                            N_GENS_ALLOWED)
from pcgsepy.lsystem.rules import StochasticRules
from pcgsepy.lsystem.solution import CandidateSolution
from pcgsepy.mapelites.behaviors import (BehaviorCharacterization, avg_ma,
                                         mame, mami, symmetry)
from pcgsepy.mapelites.bin import MAPBin
from pcgsepy.mapelites.emitters import (ContextualBanditEmitter, GreedyEmitter,
                                        HumanEmitter, HumanPrefMatrixEmitter,
                                        PreferenceBanditEmitter, RandomEmitter)
from pcgsepy.mapelites.map import MAPElites
from tqdm import trange

from pcgsepy.xml_conversion import structure_xml_converter


class DashLoggerHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)
        self.queue = []

    def emit(self, record):
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = self.format(record)
        self.queue.append(f'[{t}]\t{msg}')


logger = logging.getLogger('dash-msgs')
logger.setLevel(logging.DEBUG)
dashLoggerHandler = DashLoggerHandler()
logger.addHandler(dashLoggerHandler)


hm_callback_props = {}
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
gen_counter: int = 0
selected_bins: List[Tuple[int, int]] = []
exp_n: int = 0
rngseed: int = 42
time_elapsed = []
n_spaceships_inspected = []
current_mapelites = None
my_emitterslist = ['mapelites_random.json',
                       'mapelites_prefmatrix.json',
                       'mapelites_prefbandit.json',
                       'mapelites_contbandit.json']


def resource_path(relative_path):
# get absolute path to resource
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


app = dash.Dash(__name__,
                title='SE ICMAP-Elites',
                # external_stylesheets=[dbc.themes.CYBORG],
                external_stylesheets=[dbc.themes.DARKLY],
                # external_stylesheets=[dbc.themes.FLATLY],
                # external_stylesheets=[dbc.themes.LUMEN],
                # external_stylesheets=[dbc.themes.MORPH],
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


def set_app_layout(behavior_descriptors_names,
                   mapelites: Optional[MAPElites] = None,
                   dev_mode: bool = True):
    
    global current_mapelites
    global rngseed
    
    description_str, help_str = '', ''
    with open('./assets/description.md', 'r') as f:
        description_str = f.read()
    help_file = './assets/help_dev.md' if dev_mode else './assets/help_user.md'
    with open(help_file, 'r') as f:
        help_str = f.read()
    
    rngseed = random.randint(0, 128)
    if mapelites is None:
        random.seed(rngseed)
        random.shuffle(my_emitterslist)
        with open(my_emitterslist[0], 'r') as f:
            mapelites = json_loads(f.read())
    current_mapelites = mapelites
    
    logging.getLogger('dash-msgs').info(msg=f'Your ID is {str(rngseed).zfill(3)}; please remember this!')
    
    header = html.Div(children=[
            html.H1(children='🚀Space Engineers🚀 IC MAP-Elites',
                    className='title'),
            dcc.Markdown(children=description_str,
                         className='page-description'),
        ],
            className='header')
    
    footer = html.Div(children=[
            html.H4(children='Help',
                    className='title'),
            dcc.Markdown(help_str,
                         className='page-description')
        ],
            className='footer')
    
    mapelites_heatmap = html.Div(children=[
        dcc.Graph(id="heatmap-plot",
                  figure=go.Figure(data=[]))
        ])
    
    mapelites_controls = html.Div(
        children=[
            html.H6(children='Plot settings',
                    className='section-title'),
            html.P(children='Choose which population to display.',
                   className='generic-description'),
            dbc.DropdownMenu(label='Feasible',
                            children=[
                                dbc.DropdownMenuItem('Feasible'),
                                dbc.DropdownMenuItem('Infeasible'),
                            ],
                            id='population-dropdown'),
            html.Br(),
            html.P(children='Choose which metric to plot.',
                className='generic-description'),
            dbc.DropdownMenu(label='Fitness',
                            children=[
                                dbc.DropdownMenuItem('Fitness'),
                                dbc.DropdownMenuItem('Age'),
                                dbc.DropdownMenuItem('Coverage'),
                            ],
                            id='metric-dropdown'),
            html.Br(),
            html.P(children='Choose whether to compute the metric for the entire bin population or just the elite.',
                className='generic-description'),
            dbc.RadioItems(id='method-radio',
                        options=[
                            {'label': 'Population', 'value': 'Population'},
                            {'label': 'Elite', 'value': 'Elite'}
                        ],
                        value='Population')
            ])
    
    content_plot = html.Div(children=[
        dcc.Graph(id="content-plot",
                  figure=go.Figure(data=[])),
        ])
    
    content_properties = html.Div(
        children=[
            html.H6('Content properties',
                    className='section-title'),
            html.Div(children=[
                html.P(children='',
                       id='spaceship-size'),
                html.P(children='',
                       id='n-blocks'),
                html.P(children='Content string: '),
                dbc.Textarea(id='content-string',
                             value='',
                             contentEditable=False,
                             disabled=True),
                html.Div(children=[
                    dbc.Button('Download content',
                               id='download-btn',
                               disabled=True),
                    dcc.Download(id='download-content')
                    ])
                ])
            ])
    
    experiment_settings = html.Div(
        children=[
            html.H6(children='Experiment settings',
                    className='section-title'),
            html.Br(),
            html.Div(children=[
                html.P(children='Valid bins are: ',
                       id='valid-bins'),
                html.P(children=f'Current generation: {gen_counter}',
                       id='gen-display'),
                html.P(children=f'Selected bin(s): {selected_bins}',
                       id='selected-bin')
                ]),
            html.Br(),
            dbc.InputGroup(children=[
                dbc.InputGroupText('Feature descriptors (X, Y):'),
                dbc.DropdownMenu(label=current_mapelites.b_descs[0].name,
                             children=[dbc.DropdownMenuItem(x) for x in behavior_descriptors_names],
                             id='b0-dropdown'),
                dbc.DropdownMenu(label=current_mapelites.b_descs[1].name,
                             children=[dbc.DropdownMenuItem(x) for x in behavior_descriptors_names],
                             id='b1-dropdown')
                ],
                           className="mb-3"),
            dbc.InputGroup(children=[
                dbc.InputGroupText('Toggle L-system modules:'),
                dbc.Checklist(id='lsystem-modules',
                              options=[{'label': x.name, 'value': x.name} for x in current_mapelites.lsystem.modules],
                              value=[x.name for x in current_mapelites.lsystem.modules if x.active],
                              inline=True,
                              switch=True)
                ],
                           className="mb-3"),
            dbc.InputGroup(children=[
                dbc.InputGroupText('Fitness weights:'),
                html.Div(children=[
                    html.Div(children=[
                        dbc.Label(children=f.name),
                        html.Div(children=[
                            dcc.Slider(min=0,
                                       max=1,
                                       step=0.1,
                                       value=1,
                                       marks=None,
                                       tooltip={"placement": "bottom",
                                                "always_visible": True},
                                       id={'type': 'fitness-sldr',
                                           'index': i})
                        ],
                                 )
                        ]) for i, f in enumerate(current_mapelites.feasible_fitnesses)
                    ])
                ],
                           className="mb-3"),
            dbc.InputGroup(children=[
                dbc.InputGroupText('Select emitter:'),
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
                           style={'content-visibility': 'hidden' if not dev_mode else 'visible'}),
            dbc.InputGroup(children=[
                dbc.InputGroupText('Enforce symmetry:'),
                dbc.DropdownMenu(label='None',
                             children=[
                                 dbc.DropdownMenuItem('None'),
                                 dbc.DropdownMenuItem('X-axis'),
                                 dbc.DropdownMenuItem('Y-axis'),
                                 dbc.DropdownMenuItem('Z-axis'),
                             ],
                             id='symmetry-dropdown'),
                dbc.RadioItems(id='symmetry-radio',
                        options=[
                            {'label': 'Upper', 'value': 'Upper'},
                            {'label': 'Lower', 'value': 'Lower'}
                        ],
                        value='Upper')
                ],
                           className="mb-3"),
            dbc.InputGroup(children=[
                dbc.InputGroupText('Save/load population:'),
                dbc.Button(id='popdownload-btn',
                           children='Download current population'),
                dcc.Upload(
                    id='popupload-data',
                    children='Upload population',
                    multiple=False
                    ),
                ],
                           className="mb-3",
                           style={'content-visibility': 'hidden' if not dev_mode else 'visible'})
            ])
    
    experiment_controls = html.Div(
        children=[
            html.H6('Experiment controls'),
            html.Br(),
            dbc.Button(id='step-btn',
                       children='Apply step'),
            html.Br(),
            dbc.Button(id='reset-btn',
                       children='Initialize/Reset',
                       style={'content-visibility': 'hidden' if not dev_mode else 'visible'}),
            html.Br(),
            dbc.Button(id='selection-clr-btn',
                       children='Clear selection'),
            html.Br(),
            dbc.Button(id='selection-btn',
                       children='Toggle single bin selection'),
            html.Br(),
            dbc.Button(id='subdivide-btn',
                       children='Subdivide selected bin(s)',
                       style={'content-visibility': 'hidden' if not dev_mode else 'visible'}),
            html.Br(),
            html.Div(children=[
                dbc.Button(id='download-mapelites-btn',
                           children='Download MAP-Elites',
                           disabled=True),
                dcc.Download(id='download-mapelites')
            ],
                     style={'content-visibility': 'hidden' if not dev_mode else 'visible'})
        ])
    
    rules = html.Div(
        children=[
            html.H6(children='High-level rules'),
            dbc.Textarea(id='hl-rules',
                         value=str(current_mapelites.lsystem.hl_solver.parser.rules),
                         wrap=False),
            dbc.Button(children='Update high-level rules',
                       id='update-rules-btn')
            ],
        style={'content-visibility': 'hidden' if not dev_mode else 'visible'})
    
    log = html.Div(
        children=[
            dcc.Interval(id='interval1',
                         interval=1 * 1000,
                         n_intervals=0),
            html.H6(children='Log'),
            dbc.Textarea(id='console-out',
                         value='',
                         wrap=False,
                         contentEditable=False,
                         disabled=True)
            ])
    
    app.layout = dbc.Container(
        children=[
            header,
            html.Br(),
            dbc.Row(children=[
                dbc.Col(mapelites_heatmap, width=3),
                dbc.Col(content_plot, width=6),
                dbc.Col(content_properties, width=2)],
                    align="center"),
            dbc.Row(children=[
                dbc.Col(mapelites_controls, width=3),
                dbc.Col(experiment_settings, width=4),
                dbc.Col(experiment_controls, width=3),
                dbc.Col(rules, width=2)],
                    align="start"),
            dbc.Row(children=[
                dbc.Col(log, width={'size': 4, 'offset': 4})],
                    align="end"),
            html.Br(),
            footer
            ],
        fluid=True)


@app.callback(Output('console-out', 'value'),
              Input('interval1', 'n_intervals'))
def update_output(n):
    return ('\n'.join(dashLoggerHandler.queue))


def _from_bc_to_idx(bcs: Tuple[float, float],
                    mapelites: MAPElites) -> Tuple[int, int]:
    b0, b1 = bcs
    i = np.digitize([b0],
                    np.cumsum([0] + mapelites.bin_sizes[0]
                              [:-1]) + mapelites.b_descs[0].bounds[0],
                    right=False)[0] - 1
    j = np.digitize([b1],
                    np.cumsum([0] + mapelites.bin_sizes[1]
                              [:-1]) + mapelites.b_descs[1].bounds[0],
                    right=False)[0] - 1
    return (i, j)


def _switch(ls: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    res = []
    for e in ls:
        res.append((e[1], e[0]))
    return res


def _get_valid_bins(mapelites: MAPElites):
    valid_bins = [x.bin_idx for x in mapelites._valid_bins()]
    return _switch(valid_bins)


def format_bins(mapelites: MAPElites,
                bins_idx_list: List[Tuple[int, int]],
                str_prefix: str,
                do_switch: bool = True,
                filter_out_empty: bool = True) -> Tuple[List[Tuple[int, int]], str]:
    bins_list: List[MAPBin] = [mapelites.bins[j, i] if do_switch else mapelites.bins[i, j] for (i, j) in bins_idx_list]
    sel_bins_str = f'{str_prefix}'
    for b in bins_list:
        if filter_out_empty:
            if b.non_empty(pop='feasible') or b.non_empty(pop='infeasible'):
                i, j = b.bin_idx
                i, j = (j, i) if do_switch else (i, j)
                bc1 = np.sum([mbin.bin_size[0] for mbin in mapelites.bins[:i, j]])
                bc2 = np.sum([mbin.bin_size[1] for mbin in mapelites.bins[i, :j]])
                sel_bins_str += f' {(i, j)} [{bc1}:{bc2}];'
            elif b.bin_idx in bins_idx_list:
                bins_idx_list.remove((i, j))
        else:
            i, j = b.bin_idx
            bc1 = np.sum([mbin.bin_size[0] for mbin in mapelites.bins[:i, j]])
            bc2 = np.sum([mbin.bin_size[1] for mbin in mapelites.bins[i, :j]])
            sel_bins_str += f' {(i, j)} [{bc1}:{bc2}];'
    return bins_idx_list, sel_bins_str


def _build_heatmap(mapelites: MAPElites,
                   pop_name: str,
                   metric_name: str,
                   method_name: str) -> go.Figure:
    valid_bins = [x.bin_idx for x in mapelites._valid_bins()]    
    metric = hm_callback_props['metric'][metric_name]
    use_mean = hm_callback_props['method'][method_name]
    population = hm_callback_props['pop'][pop_name]
    # build hotmap
    disp_map = np.zeros(shape=mapelites.bins.shape)
    text = []
    for i in range(mapelites.bins.shape[0]):
        for j in range(mapelites.bins.shape[1]):
            v = mapelites.bins[i, j].get_metric(metric=metric['name'],
                                                use_mean=use_mean,
                                                population=population)
            disp_map[i, j] = v
            s = str((j, i)) if (i, j) in valid_bins else ''
            if j == 0:
                text.append([s])
            else:
                text[-1].append(s)
    # plot
    x_labels = np.cumsum([0] + mapelites.bin_sizes[0]
                         [:-1]) + mapelites.b_descs[0].bounds[0]
    y_labels = np.cumsum([0] + mapelites.bin_sizes[1]
                         [:-1]) + mapelites.b_descs[1].bounds[0]
    title = f'{pop_name} population {metric_name.lower()} ({"Average" if use_mean else "Elite"})'
    heatmap = go.Figure(data=go.Heatmap(
        z=disp_map,
        zmin=0,
        zmax=hm_callback_props['metric'][metric_name]['zmax'][population],
        x=x_labels,
        y=y_labels,
        hoverongaps=False,
        colorscale=hm_callback_props['metric'][metric_name]['colorscale'],
        text=text,
        texttemplate="%{text}",
        textfont={"color": 'rgba(0, 255, 0, 0.5)'},
    ))
    heatmap.update_xaxes(title=dict(text=mapelites.b_descs[0].name))
    heatmap.update_yaxes(title=dict(text=mapelites.b_descs[1].name))
    heatmap.update_coloraxes(colorbar_title_text=metric_name)
    heatmap.update_layout(title=dict(text=title),
                          autosize=False,
                          clickmode='event+select',
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    hovertemplate = f'{mapelites.b_descs[0].name}: X<br>{mapelites.b_descs[1].name}: Y<br>{metric_name}: Z<extra></extra>'
    hovertemplate = hovertemplate.replace(
        'X', '%{x}').replace('Y', '%{y}').replace('Z', '%{z}')
    heatmap.update_traces(hovertemplate=hovertemplate,
                          selector=dict(type='heatmap'))
    heatmap.update_layout(
        xaxis={
            # 'tickmode': 'linear',
            # 'tick0': 0,
            # 'dtick': mapelites.bin_sizes[0],
            'tickvals': x_labels
        },
        yaxis={
            # 'tickmode': 'linear',
            # 'tick0': 0,
            # 'dtick': mapelites.bin_sizes[1],
            'tickvals': y_labels
        }
    )

    return heatmap


def _get_colour_mapping(block_types: List[str]) -> Dict[str, str]:
    colour_map = {}
    for block_type in block_types:
        c = block_to_colour.get(block_type, '#ff0000')
        if block_type not in colour_map.keys():
            colour_map[block_type] = c
    return colour_map


def _get_elite_content(mapelites: MAPElites,
                       bin_idx: Tuple[int, int],
                       pop: List[CandidateSolution]) -> go.Scatter3d:
    # get elite content
    elite = mapelites.get_elite(bin_idx=bin_idx,
                                pop=pop)
    structure = elite.content
    content = structure.as_grid_array()
    arr = np.nonzero(content)
    x, y, z = arr
    cs = [content[i, j, k] for i, j, k in zip(x, y, z)]
    ss = [structure._clean_label(structure.ks[v - 1]) for v in cs]
    fig = px.scatter_3d(x=x,
                        y=y,
                        z=z,
                        color=ss,
                        color_discrete_map=_get_colour_mapping(ss),
                        labels={
                            'x': 'x',
                            'y': 'y',
                            'z': 'z',
                            'color': 'Block type'
                        },
                        title='Last clicked elite content')
    
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
                      plot_bgcolor='rgba(0,0,0,0)')
    return fig


def _apply_step(mapelites: MAPElites,
                selected_bins: List[Tuple[int, int]],
                gen_counter: int) -> bool:
    if len(selected_bins) > 0:
        valid = True
        if mapelites.enforce_qnt:
            valid_bins = [list(x.bin_idx) for x in mapelites._valid_bins()]
            for bin_idx in selected_bins:
                valid &= bin_idx in valid_bins
        if valid:
            logging.getLogger('dash-msgs').info(msg=f'Started step {gen_counter + 1}...')
            mapelites._interactive_step(bin_idxs=selected_bins,
                                        gen=gen_counter)
            logging.getLogger('dash-msgs').info(msg=f'Completed step {gen_counter + 1} (created {mapelites.n_new_solutions} solutions); running {N_EMITTER_STEPS} additional emitter steps if available...')
            mapelites.n_new_solutions = 0
            with trange(N_EMITTER_STEPS, desc='Emitter steps: ') as iterations:
                for _ in iterations:
                    mapelites.emitter_step(gen=gen_counter)
            logging.getLogger('dash-msgs').info(msg=f'Emitter step(s) completed (created {mapelites.n_new_solutions} solutions).')
            mapelites.n_new_solutions = 0
            return True
        else:
            logging.getLogger('dash-msgs').info(msg='Step not applied: invalid bin(s) selected.')
            return False


def _apply_reset(mapelites: MAPElites) -> bool:
    logging.getLogger('dash-msgs').info(msg='Started resetting all bins (this may take a while)...')
    mapelites.reset()
    logging.getLogger('dash-msgs').info(msg='Reset completed.')
    return True


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


def _apply_bc_change(bcs: Tuple[str, str],
                     mapelites: MAPElites) -> bool:
    b0, b1 = bcs
    logging.getLogger('dash-msgs').info(msg=f'Updating feature descriptors to ({b0}, {b1})...')
    b0 = behavior_descriptors[[b.name for b in behavior_descriptors].index(b0)]
    b1 = behavior_descriptors[[b.name for b in behavior_descriptors].index(b1)]
    mapelites.update_behavior_descriptors((b0, b1))
    logging.getLogger('dash-msgs').info(msg='Feature descriptors update completed.')
    return True


def _apply_bin_subdivision(mapelites: MAPElites,
                           selected_bins: List[Tuple[int, int]]) -> bool:
    bin_idxs = [(x[1], x[0]) for x in selected_bins]
    for bin_idx in bin_idxs:
        mapelites.subdivide_range(bin_idx=bin_idx)
    logging.getLogger('dash-msgs').info(msg=f'Subdivided bin(s): {selected_bins}.')
    return True


def _apply_modules_update(mapelites: MAPElites,
                          modules: List[str]) -> bool:
    all_modules = [x for x in mapelites.lsystem.modules]
    names = [x.name for x in all_modules]
    for i, module in enumerate(names):
        if module in modules and not all_modules[i].active:
            # activate module
            mapelites.toggle_module_mutability(module=module)
            logging.getLogger('dash-msgs').info(msg=f'Enabled {module}.')
            break
        elif module not in modules and all_modules[i].active:
            # deactivate module
            mapelites.toggle_module_mutability(module=module)
            logging.getLogger('dash-msgs').info(msg=f'Disabled {module}.')
            break
    return True


def _apply_rules_update(mapelites: MAPElites,
                        rules: str) -> bool:
    new_rules = StochasticRules()
    for rule in rules.split('\n'):
        lhs, p, rhs = rule.strip().split(' ')
        new_rules.add_rule(lhs=lhs,
                           rhs=rhs,
                           p=float(p))
    try:
        new_rules.validate()
        mapelites.lsystem.hl_solver.parser.rules = new_rules
        logging.getLogger('dash-msgs').info(msg=f'L-system rules updated.')
        return True
    except AssertionError as e:
        logging.getLogger('dash-msgs').info(msg=f'Failed updating L-system rules ({e}).')
        return False


def _apply_fitness_reweight(mapelites: MAPElites,
                            weights: List[float]) -> bool:
    mapelites.update_fitness_weights(weights=weights)
    logging.getLogger('dash-msgs').info(msg='Updated fitness functions weights.')
    hm_callback_props['metric']['Fitness']['zmax']['feasible'] = sum([x.weight * x.bounds[1] for x in mapelites.feasible_fitnesses]) + mapelites.nsc
    return True


def _apply_bin_selection_toggle(mapelites: MAPElites) -> bool:
    mapelites.enforce_qnt = not mapelites.enforce_qnt
    logging.getLogger('dash-msgs').info(msg=f'MAP-Elites single bin selection set to {mapelites.enforce_qnt}.')
    
    
def _apply_emitter_change(mapelites: MAPElites,
                          emitter_name: str) -> bool:
    if emitter_name == 'Random':
        mapelites.emitter = RandomEmitter()
        logging.getLogger('dash-msgs').info(msg=f'Emitter set to {emitter_name}')
        return True
    if emitter_name == 'Greedy':
        mapelites.emitter = GreedyEmitter()
        logging.getLogger('dash-msgs').info(msg=f'Emitter set to {emitter_name}')
        return True
    elif emitter_name == 'Preference-matrix':
        mapelites.emitter = HumanPrefMatrixEmitter()
        mapelites.emitter._build_pref_matrix(bins=mapelites.bins)
        logging.getLogger('dash-msgs').info(msg=f'Emitter set to {emitter_name}')
        return True
    elif emitter_name == 'Contextual Bandit':
        mapelites.emitter = ContextualBanditEmitter()
        logging.getLogger('dash-msgs').info(msg=f'Emitter set to {emitter_name}')
        return True
    elif emitter_name == 'Preference Bandit':
        mapelites.emitter = PreferenceBanditEmitter()
        logging.getLogger('dash-msgs').info(msg=f'Emitter set to {emitter_name}')
        return True
    elif emitter_name == 'None':
        mapelites.emitter = HumanEmitter()
        logging.getLogger('dash-msgs').info(msg=f'Emitter set to {emitter_name}')
        return True
    else:
        logging.getLogger('dash-msgs').info(msg=f'Unrecognized emitter type {emitter_name}')
        return False

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
    return dict(content=json_dumps(current_mapelites), filename=f'{fname}.json')


@app.callback(Output('heatmap-plot', 'figure'),
              Output('content-plot', 'figure'),
              Output('valid-bins', 'children'),
              Output('gen-display', 'children'),
              Output('hl-rules', 'value'),
              Output('selected-bin', 'children'),
              Output('content-string', 'value'),
              Output('spaceship-size', 'children'),
              Output('n-blocks', 'children'),
              Output('download-mapelites-btn', 'disabled'),
              Output('download-btn', 'disabled'),
              Output('step-btn', 'disabled'),
              Output("download-content", "data"),
    
              State('heatmap-plot', 'figure'),
              State('hl-rules', 'value'),
              State('content-plot', 'figure'),
              State('content-string', 'value'),
              State('spaceship-size', 'children'),
              State('n-blocks', 'children'),
              
              Input('population-dropdown', 'label'),
              Input('metric-dropdown', 'label'),
              Input('method-radio', 'value'),
              Input('step-btn', 'n_clicks'),
              Input('reset-btn', 'n_clicks'),
              Input('subdivide-btn', 'n_clicks'),
              Input({'type': 'fitness-sldr', 'index': ALL}, 'value'),
              Input('b0-dropdown', 'label'),
              Input('b1-dropdown', 'label'),
              Input('lsystem-modules', 'value'),
              Input('update-rules-btn', 'n_clicks'),
              Input('heatmap-plot', 'clickData'),
              Input('selection-btn', 'n_clicks'),
              Input('selection-clr-btn', 'n_clicks'),
              Input('emitter-dropdown', 'label'),
              Input("download-btn", "n_clicks"),
              Input('popdownload-btn', 'n_clicks'),
              Input('popupload-data', 'contents'),
              Input('symmetry-dropdown', 'label'),
              Input('symmetry-radio', 'value'),
              )
def general_callback(curr_heatmap, rules, curr_content, cs_string, cs_size, cs_n_blocks,
                     pop_name, metric_name, method_name, n_clicks_step, n_clicks_reset, n_clicks_sub, weights, b0, b1, modules, n_clicks_rules, clickData, selection_btn, clear_btn, emitter_name, n_clicks_cs_download, n_clicks_popdownload, upload_contents, symm_axis, symm_orientation):
    content_dl = None
    global current_mapelites
    global gen_counter
    global my_emitterslist
    global selected_bins
    global exp_n
    global rngseed
    global n_spaceships_inspected
    global time_elapsed
    
    
    ctx = dash.callback_context

    if not ctx.triggered:
        event_trig = None
    else:
        event_trig = ctx.triggered[0]['prop_id'].split('.')[0]

    if event_trig == 'step-btn':
        s = time.time()
        res = _apply_step(mapelites=current_mapelites,
                          selected_bins=[[x[1], x[0]] for x in selected_bins],
                          gen_counter=gen_counter)
        if res:
            elapsed = time.time() - s
            gen_counter += 1
            curr_heatmap = _build_heatmap(mapelites=current_mapelites,
                                          pop_name=pop_name,
                                          metric_name=metric_name,
                                          method_name=method_name)
            
            n_spaceships_inspected[-1].append(1)  # new generation of last experiment
            
            if time_elapsed == []:  # first generation of first experiment
                time_elapsed.append([elapsed])
            elif time_elapsed[-1] == []:  # first generation of latest experiment
                time_elapsed[-1] = [elapsed]
            else:  # latest generation of latest experiment
                time_elapsed[-1].append(elapsed)
            
    elif event_trig == 'reset-btn':
        res = _apply_reset(mapelites=current_mapelites)
        if res:
            gen_counter = 0
            curr_heatmap = _build_heatmap(mapelites=current_mapelites,
                                          pop_name=pop_name,
                                          metric_name=metric_name,
                                          method_name=method_name)
            
            if n_spaceships_inspected == []:
                n_spaceships_inspected.append([0])
            else:
                n_spaceships_inspected[-1] = [[0]]
            
            if time_elapsed == []:
                time_elapsed.append([])
            else:
                time_elapsed[-1] = [[]]
            
    elif event_trig == 'b0-dropdown' or event_trig == 'b1-dropdown':
        res = _apply_bc_change(bcs=(b0, b1),
                               mapelites=current_mapelites)
        if res:
            curr_heatmap = _build_heatmap(mapelites=current_mapelites,
                                          pop_name=pop_name,
                                          metric_name=metric_name,
                                          method_name=method_name)
    elif event_trig == 'subdivide-btn':
        res = _apply_bin_subdivision(mapelites=current_mapelites,
                                     selected_bins=selected_bins)
        if res:
            curr_heatmap = _build_heatmap(mapelites=current_mapelites,
                                          pop_name=pop_name,
                                          metric_name=metric_name,
                                          method_name=method_name)
            selected_bins = []
    elif event_trig == 'lsystem-modules':
        res = _apply_modules_update(mapelites=current_mapelites,
                                    modules=modules)
    elif event_trig == 'update-rules-btn':
        res = _apply_rules_update(mapelites=current_mapelites,
                                  rules=rules)
    # event_trig is a str of a dict, ie: '{"index":*,"type":"fitness-sldr"}', go figure
    elif event_trig is not None and 'fitness-sldr' in event_trig:
        res = _apply_fitness_reweight(mapelites=current_mapelites,
                                      weights=weights)
        if res:
            curr_heatmap = _build_heatmap(mapelites=current_mapelites,
                                     pop_name=pop_name,
                                     metric_name=metric_name,
                                     method_name=method_name)
    elif event_trig == 'population-dropdown' or event_trig == 'metric-dropdown' or event_trig == 'method-radio':
        curr_heatmap = _build_heatmap(mapelites=current_mapelites,
                                     pop_name=pop_name,
                                     metric_name=metric_name,
                                     method_name=method_name)
    elif event_trig == 'symmetry-dropdown' or event_trig == 'symmetry-radio':
        logging.getLogger('dash-msgs').info(msg=f'Updating all solutions to enforce symmetry...')
        current_mapelites.reassign_all_content(sym_axis=symm_axis[0].lower() if symm_axis != "None" else None,
                                               sym_upper=symm_orientation == 'Upper')
        curr_content = go.Figure(data=[])
        cs_string = cs_size = cs_n_blocks  = ''
        logging.getLogger('dash-msgs').info(msg=f'Symmetry enforcement completed.')
    elif event_trig == 'heatmap-plot' or event_trig == 'population_dropdown':
        i, j = _from_bc_to_idx(bcs=(clickData['points'][0]['x'],
                                    clickData['points'][0]['y']),
                               mapelites=current_mapelites)
        if current_mapelites.bins[j, i].non_empty(pop='feasible' if pop_name == 'Feasible' else 'infeasible'):
            curr_content = _get_elite_content(mapelites=current_mapelites,
                                              bin_idx=(j, i),
                                              pop='feasible' if pop_name == 'Feasible' else 'infeasible')
            
            if n_spaceships_inspected == []:  # first experiment
                n_spaceships_inspected.append([1])  # one spaceship selected in the first generation
            else:
                n_spaceships_inspected[-1][-1] += 1  # update latest generation of the latest experiment
            
            if not current_mapelites.enforce_qnt and selected_bins != []:
                if [i, j] not in selected_bins:
                    selected_bins.append([i, j])
                else:
                    selected_bins.remove([i, j])
            else:
                selected_bins = [[i, j]]
            cs_string = cs_size = cs_n_blocks = ''
            if len(selected_bins) > 0:
                elite = current_mapelites.get_elite(bin_idx=_switch([selected_bins[-1]])[0],
                                            pop='feasible' if pop_name == 'Feasible' else 'infeasible')
                cs_string = elite.string
                cs_size = f'Spaceship size: {elite.size}'
                cs_n_blocks = f'Number of blocks: {elite.n_blocks}'
        else:
            logging.getLogger('dash-msgs').info(msg=f'Empty bin selected ({i}, {j}).')
    elif event_trig == 'selection-btn':
        _ = _apply_bin_selection_toggle(mapelites=current_mapelites)
        if current_mapelites.enforce_qnt and selected_bins:
            selected_bins = [selected_bins[-1]]
    elif event_trig == 'selection-clr-btn':
        logging.getLogger('dash-msgs').info(msg='Cleared bins selection.')
        selected_bins = []
        curr_content = go.Figure(data=[])
        cs_string = cs_size = cs_n_blocks  = ''
    elif event_trig == 'emitter-dropdown':
        _ = _apply_emitter_change(mapelites=current_mapelites,
                                  emitter_name=emitter_name)
    elif event_trig == 'download-btn':
        if cs_string != '':
            exp_n += 1
            def write_archive(bytes_io):
                with ZipFile(bytes_io, mode="w") as zf:
                    with open('./assets/thumb.png', 'rb') as f:
                        thumbnail_img = f.read()
                    zf.writestr('thumb.png', thumbnail_img)
                    elite = current_mapelites.get_elite(bin_idx=_switch([selected_bins[-1]])[0],
                                            pop='feasible' if pop_name == 'Feasible' else 'infeasible')
                    zf.writestr('bp.sbc', structure_xml_converter(structure=elite.content, name=f'My Spaceship ({rngseed}) (exp{exp_n})'))
                    zf.writestr(f'spaceship_{rngseed}_exp{exp_n}', cs_string)
            content_dl = dcc.send_bytes(write_archive, f'MySpaceship_{rngseed}_exp{exp_n}.zip')
            if exp_n >= len(my_emitterslist):
                curr_heatmap = go.Figure(data=[])
                selected_bins = []
                curr_content = go.Figure(data=[])
                cs_string = cs_size = cs_n_blocks  = ''
                content_dl = dict(content=json.dumps({
                    'time_elapsed': time_elapsed,
                    'n_interactions': n_spaceships_inspected
                    }),
                                  filename=f'user_metrics_{rngseed}')   
                logging.getLogger('dash-msgs').info(f'Reached end of all experiments! Please go back to the questionnaire to continue the evaluation.')
            else:
                gen_counter = 0
                # with open(my_emitterslist[exp_n], 'r') as f:
                #     current_mapelites = json_loads(f.read())
                current_mapelites.reset()
                current_mapelites.generate_initial_populations()
                curr_heatmap = _build_heatmap(mapelites=current_mapelites,
                                        pop_name=pop_name,
                                        metric_name=metric_name,
                                        method_name=method_name)
                selected_bins = []
                curr_content = go.Figure(data=[])
                cs_string = cs_size = cs_n_blocks  = ''
                n_spaceships_inspected.append([1])  # first generation of new experiment
                time_elapsed.append([])  # first generation of new experiment
                logging.getLogger('dash-msgs').info(msg=f'Reached end of experiment {exp_n}! Loaded next experiment. Fill out the questionnaire before continuing.')
    elif event_trig == 'popdownload-btn':
        content_dl = dict(content=json.dumps([b.to_json() for b in current_mapelites.bins.flatten().tolist()]),
                          filename=f'population_{rngseed}_exp{exp_n}.json')
    elif event_trig == 'popupload-data':
        _, upload_contents = upload_contents.split(',')
        upload_contents = base64.b64decode(upload_contents).decode()
        all_bins = np.asarray([MAPBin.from_json(x) for x in json.loads(upload_contents)])
        current_mapelites.reset(lcs=[])
        all_bins = all_bins.reshape(current_mapelites.bin_qnt)
        current_mapelites.bins = all_bins
        current_mapelites.reassign_all_content()
        logging.getLogger('dash-msgs').info(msg=f'Set population from file successfully.')
        curr_heatmap = _build_heatmap(mapelites=current_mapelites,
                                    pop_name=pop_name,
                                    metric_name=metric_name,
                                    method_name=method_name)
    elif event_trig is None:
        curr_heatmap = _build_heatmap(mapelites=current_mapelites,
                                    pop_name=pop_name,
                                    metric_name=metric_name,
                                    method_name=method_name)
    else:
        logging.getLogger('dash-msgs').info(msg=f'Unrecognized event trigger: {event_trig}. No operations have been applied!')

    selected_bins, selected_bins_str = format_bins(mapelites=current_mapelites,
                                                   bins_idx_list=selected_bins,
                                                   do_switch=True,
                                                   str_prefix='Selected bin(s):',
                                                   filter_out_empty=True) 
    _, valid_bins_str = format_bins(mapelites=current_mapelites,
                                    do_switch=False,
                                    bins_idx_list=_get_valid_bins(mapelites=current_mapelites),
                                    str_prefix='Valid bins are:',
                                    filter_out_empty=False)
    
    return curr_heatmap, curr_content, valid_bins_str, f'Current generation: {gen_counter}', str(current_mapelites.lsystem.hl_solver.parser.rules), selected_bins_str, cs_string, cs_size, cs_n_blocks, False, gen_counter < N_GENS_ALLOWED, gen_counter >= N_GENS_ALLOWED, content_dl
