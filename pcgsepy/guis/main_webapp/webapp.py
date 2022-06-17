import json
from datetime import datetime
from typing import Dict, List, Tuple

import dash
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, dcc, html
from dash.dependencies import Input, Output, State
from pcgsepy.common.jsonifier import json_dumps, json_loads
from pcgsepy.config import BIN_POP_SIZE, CS_MAX_AGE, N_GENS_ALLOWED
from pcgsepy.lsystem.rules import StochasticRules
from pcgsepy.lsystem.solution import CandidateSolution
from pcgsepy.mapelites.emitters import (ContextualBanditEmitter,
                                        HumanPrefMatrixEmitter, RandomEmitter)
from pcgsepy.mapelites.map import MAPElites, get_structure


class CustomLogger:
    def __init__(self) -> None:
        self.queue = []
    
    def log(self,
            msg: str) -> None:
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.queue.append(f'[{t}]\t{msg}')


hm_callback_props = {}


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
                title='SE ICMAP-Elites',
                external_stylesheets=[
                    'https://codepen.io/chriddyp/pen/bWLwgP.css'],
                update_title=None)


def set_app_layout(mapelites: MAPElites,
                   behavior_descriptors_names,
                   dev_mode: bool = True):
    description_str, help_str = '', ''
    with open('./assets/description.md', 'r') as f:
        description_str = f.read()
    help_file = './assets/help_dev.md' if dev_mode else './assets/help_user.md'
    with open(help_file, 'r') as f:
        help_str = f.read()
    
    app.layout = html.Div(children=[
        # HEADER
        html.Div(children=[
            html.H1(children='🚀Space Engineers🚀 IC MAP-Elites',
                    className='title'),
            dcc.Markdown(children=description_str,
                         className='page-description'),
        ],
            className='header'),
        html.Br(),
        # BODY
        html.Div(children=[
            # PLOTS
            html.Div(children=[
                # HEATMAP
                html.Div(children=[
                    dcc.Graph(id="heatmap-plot",
                              figure=go.Figure(data=[]))
                ],
                    className='heatmap-div'),
                # CONTENT PLOT
                html.Div(children=[
                    dcc.Graph(id="content-plot",
                              figure=go.Figure(data=[])),
                ],
                    className='content-div'),
            ],
                className='plots'),
            # PROPERTIES & DOWNLOAD
            html.Div(children=[
                html.H6('Content properties',
                        className='section-title'),
                html.Div(children=[
                    html.P(children='',
                           className='properties-text',
                           id='spaceship-size'),
                    html.P(children='',
                           className='properties-text',
                           id='n-blocks'),
                    html.P(children='Content string:',
                           className='properties-text'),
                    dcc.Textarea(id='content-string',
                                    value='',
                                    contentEditable=False,
                                    disabled=True,
                                    style={'width': '100%', 'height': 150}),
                    html.Div(children=[
                        html.Button('Download content',
                                    id='download-btn',
                                    className='button',
                                    disabled=True),
                        dcc.Download(id='download-content')
                    ],
                        className='button-div')
                ],
                    style={'padding-left': '10px'}),
            ],
                className='properties-div'),
            html.Br(),
            # PLOT CONTROLS
            html.Div(children=[
                html.H6(children='Plot settings',
                        className='section-title'),
                html.P(children='Choose which population to display.',
                       className='generic-description'),
                dcc.Dropdown(['Feasible', 'Infeasible'],
                             'Feasible',
                             id='population-dropdown',
                             className='dropdown'),
                html.Br(),
                html.P(children='Choose which metric to plot.',
                       className='generic-description'),
                dcc.Dropdown(['Fitness', 'Age', 'Coverage'],
                             'Fitness',
                             id='metric-dropdown',
                             className='dropdown'),
                html.Br(),
                html.P(children='Choose whether to compute the metric for the entire bin population or just the elite.',
                       className='generic-description'),
                dcc.RadioItems(['Population', 'Elite'],
                               'Population',
                               id='method-radio',
                               className='radio')
            ],
                className='graph-controls-div'),
            # EXPERIMENT SETTINGS
            html.Div(children=[
                html.H6(children='Experiment settings',
                        className='section-title'),
                html.Div(children=[
                    html.P(children='Valid bins are: ',
                           className='properties-text',
                           id='valid-bins'),
                    html.P(children='Current generation: 0',
                           className='properties-text',
                           id='gen-display'),
                    html.P(children='Selected bin(s): []',
                           className='properties-text',
                           id='selected-bin')
                ],
                    style={'margin-left': '10px'}),
                html.H6(children='Choose feature descriptors (X, Y):',
                        className='section-title'),
                html.Div(children=[
                    html.Div(children=[
                        dcc.Dropdown(behavior_descriptors_names,
                                     mapelites.b_descs[0].name,
                                     id='b0-dropdown',
                                     className='dropdown')
                    ],
                        style={'width': '50%'}),
                    html.Div(children=[
                        dcc.Dropdown(behavior_descriptors_names,
                                     mapelites.b_descs[1].name,
                                     id='b1-dropdown',
                                     className='dropdown')
                    ],
                        style={'width': '50%'}),
                ],
                    style={'display': 'flex', 'text-align': 'center', 'margin-left': '10px'}),
                html.H6(children='Toggle L-system modules',
                        className='section-title'),
                html.Div(children=[
                    dcc.Checklist(id='lsystem-modules',
                                  options=[
                                      x.name for x in mapelites.lsystem.modules],
                                  value=[
                                      x.name for x in mapelites.lsystem.modules if x.active],
                                  inline=True,
                                  className='checkboxes'
                                  )],
                         style={'text-align': 'center'}
                         ),
                html.H6(children='Control fitness weights',
                        className='section-title'),
                html.Div(children=[
                    html.Div(children=[
                        html.P(children=f.name,
                               className='generic-description'),
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
                        )],
                        style={'width': '80%', 'vertical-align': 'middle', 'margin': '0 auto',
                               'display': 'grid', 'grid-template-columns': '40% 60%'}
                    ) for i, f in enumerate(mapelites.feasible_fitnesses)
                ]),
                html.Div(children=[
                    html.H6(children='Select emitter',
                            className='section-title'),
                    html.Div(children=[
                        html.Div(children=[
                            dcc.Dropdown(['Random', 'Preference-matrix', 'Contextual Bandit'],
                                'Random',
                                id='emitter-dropdown',
                                className='dropdown',
                                style={'width': '100%'}),
                            ],
                                style={'width': '80%', 'vertical-align': 'middle', 'margin': '0 auto'})
                        ]),
                    ],
                         style={'content-visibility': 'hidden' if not dev_mode else 'visible'}),
            ],
                className='experiment-controls-div'),
            # EXPERIMENT CONTROLS
            html.Div(children=[
                html.H6('Experiment controls',
                        className='section-title'),
                html.Div(children=[
                    html.Button(children='Apply step',
                                id='step-btn',
                                n_clicks=0,
                                className='button')
                ],
                    className='button-div'),
                html.Br(),
                html.Div(children=[
                    html.Button(children='Initialize/Reset',
                                id='reset-btn',
                                n_clicks=0,
                                className='button')
                ],
                    className='button-div',
                    style={'content-visibility': 'hidden' if not dev_mode else 'visible'}),
                html.Br(),
                html.Div(children=[
                    html.Button(children='Clear selection',
                                id='selection-clr-btn',
                                n_clicks=0,
                                className='button')
                ],
                    className='button-div'),
                html.Br(),
                html.Div(children=[
                    html.Button(children='Toggle single bin selection',
                                id='selection-btn',
                                n_clicks=0,
                                className='button')
                ],
                    className='button-div'),
                html.Br(),
                html.Div(children=[
                    html.Button(children='Subdivide selected bin(s)',
                                id='subdivide-btn',
                                n_clicks=0,
                                className='button')
                ],
                    className='button-div',
                    style={'content-visibility': 'hidden' if not dev_mode else 'visible'}),
                html.Br(),
                html.Div(children=[
                    html.Button(children='Download MAP-Elites',
                                id='download-mapelites-btn',
                                className='button',
                                disabled=True),
                    dcc.Download(id='download-mapelites')
                ],
                    className='button-div')
            ],
                className='experiment-controls-div'),
            # RULES
            html.Div(children=[
                html.H6(children='High-level rules',
                        className='section-title'),
                dcc.Textarea(id='hl-rules',
                                value=str(
                                    mapelites.lsystem.hl_solver.parser.rules),
                                wrap='False',
                                style={'width': '100%', 'height': 250}),
                html.Div(children=[
                    html.Button(children='Update high-level rules',
                                id='update-rules-btn',
                                n_clicks=0,
                                className='button')
                ],
                    className='button-div'),
            ],
                # style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
                className='experiment-controls-div',
                style={'content-visibility': 'hidden' if not dev_mode else 'visible'}),
        ],
            className='body-div'),
        html.Br(),
        html.Div(children=[
            # LOG
            html.Div(children=[
                dcc.Interval(id='interval1',
                                interval=1 * 1000,
                                n_intervals=0),
                html.H6(children='Log',
                        className='section-title'),
                dcc.Textarea(id='console-out',
                                value='',
                                wrap='False',
                                contentEditable=False,
                                disabled=True,
                                style={'width': '100%', 'height': 300})
            ],
                style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'})
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
        # client-side storage
        dcc.Store(id='gen-counter'),
        dcc.Store(id='selected-bins'),
        dcc.Store(id='logger', data=json_dumps(obj=CustomLogger())),
        dcc.Store(id='mapelites', data=json_dumps(obj=mapelites))
    ])


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


def _get_selected_bins_json(selected_bins):
    _switch(selected_bins)


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
            disp_map[i, j] = v  # if v > 0 else None
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
            # 'dtick': mapelites.bin_sizes[0]
            'tickvals': x_labels
        },
        yaxis={
            # 'tickmode': 'linear',
            # 'tick0': 0,
            # 'dtick': mapelites.bin_sizes[1]
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
    structure = get_structure(string=elite.ll_string,
                              extra_args={
                                  'alphabet': mapelites.lsystem.ll_solver.atoms_alphabet
                            })
    # add hull
    if mapelites.hull_builder is not None:
        mapelites.hull_builder.add_external_hull(structure=structure)
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
    fig.update_layout(scene=dict(aspectmode='data'),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    return fig


def _apply_step(mapelites: MAPElites,
                selected_bins: List[Tuple[int, int]],
                gen_counter: int,
                logger: CustomLogger) -> bool:
    
    import time
    
    if len(selected_bins) > 0:
        valid = True
        if mapelites.enforce_qnt:
            valid_bins = [list(x.bin_idx) for x in mapelites._valid_bins()]
            for bin_idx in selected_bins:
                valid &= bin_idx in valid_bins
        if valid:
            logger.log(msg=f'Started step {gen_counter}...')
            mapelites._interactive_step(bin_idxs=selected_bins,
                                        gen=gen_counter)
            logger.log(msg=f'Completed step {gen_counter + 1}; running 5 additional emitter steps if available...')
            for _ in range(5):
                mapelites.emitter_step(gen=gen_counter)
            logger.log(msg=f'Emitter step(s) completed.')
            return True
        else:
            logger.log(msg='Step not applied: invalid bin(s) selected.')
            return False


def _apply_reset(mapelites: MAPElites,
                 logger: CustomLogger) -> bool:
    logger.log(msg='Started resetting all bins (this may take a while)...')
    mapelites.reset()
    logger.log(msg='Reset completed.')
    return True


def _apply_bc_change(mapelites: MAPElites,
                     logger: CustomLogger) -> bool:
    logger.log(msg=f'Updating feature descriptors to ({b0}, {b1})...')
    b0 = mapelites.b_descs[[b.name for b in mapelites.b_descs].index(b0)]
    b1 = mapelites.b_descs[[b.name for b in mapelites.b_descs].index(b1)]
    mapelites.update_behavior_descriptors((b0, b1))
    logger.log(msg='Feature descriptors update completed.')
    return True


def _apply_bin_subdivision(mapelites: MAPElites,
                           selected_bins: List[Tuple[int, int]],
                           logger: CustomLogger) -> bool:
    bin_idxs = [(x[1], x[0]) for x in selected_bins]
    for bin_idx in bin_idxs:
        mapelites.subdivide_range(bin_idx=bin_idx)
    logger.log(msg=f'Subdivided bin(s): {selected_bins}.')
    return True


def _apply_modules_update(mapelites: MAPElites,
                          modules: List[str],
                          logger: CustomLogger) -> bool:
    all_modules = [x for x in mapelites.lsystem.modules]
    names = [x.name for x in all_modules]
    for i, module in enumerate(names):
        if module in modules and not all_modules[i].active:
            # activate module
            mapelites.toggle_module_mutability(module=module)
            logger.log(msg=f'Enabled {module}.')
            break
        elif module not in modules and all_modules[i].active:
            # deactivate module
            mapelites.toggle_module_mutability(module=module)
            logger.log(msg=f'Disabled {module}.')
            break
    return True


def _apply_rules_update(mapelites: MAPElites,
                        rules: str,
                        logger: CustomLogger) -> bool:
    new_rules = StochasticRules()
    for rule in rules.split('\n'):
        lhs, p, rhs = rule.strip().split(' ')
        new_rules.add_rule(lhs=lhs,
                           rhs=rhs,
                           p=float(p))
    try:
        new_rules.validate()
        mapelites.lsystem.hl_solver.parser.rules = new_rules
        logger.log(msg=f'L-system rules updated.')
        return True
    except AssertionError as e:
        logger.log(msg=f'Failed updating L-system rules ({e}).')
        return False


def _apply_fitness_reweight(mapelites: MAPElites,
                            weights: List[float],
                                logger: CustomLogger) -> bool:
    mapelites.update_fitness_weights(weights=weights)
    logger.log(msg='Updated fitness functions weights.')
    hm_callback_props['metric']['Fitness']['zmax']['feasible'] = sum([x.weight * x.bounds[1] for x in mapelites.feasible_fitnesses]) + mapelites.nsc
    return True


def _apply_bin_selection_toggle(mapelites: MAPElites,
                                logger: CustomLogger) -> bool:
    mapelites.enforce_qnt = not mapelites.enforce_qnt
    logger.log(msg=f'MAP-Elites single bin selection set to {mapelites.enforce_qnt}.')
    
    
def _apply_emitter_change(mapelites: MAPElites,
                          emitter_name: str,
                          logger: CustomLogger) -> bool:
    if emitter_name == 'Random':
        mapelites.emitter = RandomEmitter()
        logger.log(msg=f'Emitter set to {emitter_name}')
        return True
    elif emitter_name == 'Preference-matrix':
        mapelites.emitter = HumanPrefMatrixEmitter()
        mapelites.emitter._build_pref_matrix(bins=mapelites.bins)
        logger.log(msg=f'Emitter set to {emitter_name}')
        return True
        pass
    elif emitter_name == 'Contextual Bandit':
        mapelites.emitter = ContextualBanditEmitter()
        logger.log(msg=f'Emitter set to {emitter_name}')
        return True
        pass
    else:
        logger.log(msg=f'Unrecognized emitter type {emitter_name}')
        return False


@app.callback(Output('console-out', 'value'),
              Input('interval1', 'n_intervals'),
              Input('logger', 'data'))
def update_output(n, logger):
    logger: CustomLogger = json_loads(logger)
    return ('\n'.join(logger.queue))


@app.callback(
    Output("download-content", "data"),
    Input("download-btn", "n_clicks"),
    State('content-string', 'value'),
    prevent_initial_call=True,
)
def download_content(n_clicks,
                     content_string):
    if content_string != '':
        return dict(content=content_string, filename='MySpaceship.txt')


@app.callback(
    Output("download-mapelites", "data"),
    Input("download-mapelites-btn", "n_clicks"),
    State('mapelites', 'data'),
    State('gen-counter', 'data'),
    prevent_initial_call=True,
)
def download_mapelites(n_clicks,
                       mapelites,
                       gen_counter):
    if mapelites != '':
        me: MAPElites = json_loads(s=mapelites)
        t = datetime.now().strftime("%Y%m%d%H%M%S")
        fname = f'{t}_mapelites_{me.emitter.name}_gen{str(gen_counter).zfill(2)}'
        return dict(content=mapelites, filename=f'{fname}.json')


@app.callback(Output('heatmap-plot', 'figure'),
              Output('content-plot', 'figure'),
              Output('valid-bins', 'children'),
              Output('gen-display', 'children'),
              Output('gen-counter', 'data'),
              Output('mapelites', 'data'),
              Output('hl-rules', 'value'),
              Output('selected-bin', 'children'),
              Output('selected-bins', 'data'),
              Output('content-string', 'value'),
              Output('spaceship-size', 'children'),
              Output('n-blocks', 'children'),
              Output('logger', 'data'),
              Output('download-mapelites-btn', 'disabled'),
              Output('download-btn', 'disabled'),
              State('heatmap-plot', 'figure'),
              State('selected-bins', 'data'),
              State('gen-counter', 'data'),
              State('mapelites', 'data'),
              State('hl-rules', 'value'),
              State('selected-bin', 'children'),
              State('content-plot', 'figure'),
              State('content-string', 'value'),
              State('spaceship-size', 'children'),
              State('n-blocks', 'children'),
              State('logger', 'data'),
              Input('population-dropdown', 'value'),
              Input('metric-dropdown', 'value'),
              Input('method-radio', 'value'),
              Input('step-btn', 'n_clicks'),
              Input('reset-btn', 'n_clicks'),
              Input('subdivide-btn', 'n_clicks'),
              Input({'type': 'fitness-sldr', 'index': ALL}, 'value'),
              Input('b0-dropdown', 'value'),
              Input('b1-dropdown', 'value'),
              Input('lsystem-modules', 'value'),
              Input('update-rules-btn', 'n_clicks'),
              Input('heatmap-plot', 'clickData'),
              Input('selection-btn', 'n_clicks'),
              Input('selection-clr-btn', 'n_clicks'),
              Input('emitter-dropdown', 'value'))
def general_callback(curr_heatmap, selected_bins, gen_counter, mapelites, rules,
                     curr_selected, curr_content, cs_string, cs_size, cs_n_blocks, logger,
                     pop_name, metric_name, method_name, n_clicks_step, n_clicks_reset, n_clicks_sub, weights, b0, b1, modules, n_clicks_rules,
                     clickData, selection_btn, clear_btn, emitter_name):
    gen_counter: int = json.loads(gen_counter) if gen_counter else 0
    selected_bins = json.loads(selected_bins) if selected_bins else []
    logger: CustomLogger = json_loads(logger)
    mapelites: MAPElites = json_loads(s=mapelites)
    
    ctx = dash.callback_context

    if not ctx.triggered:
        event_trig = None
    else:
        event_trig = ctx.triggered[0]['prop_id'].split('.')[0]

    if event_trig == 'step-btn':
        res = _apply_step(mapelites=mapelites,
                          selected_bins=[[x[1], x[0]] for x in selected_bins],
                          gen_counter=gen_counter,
                          logger=logger)
        if res:
            gen_counter += 1
            curr_heatmap = _build_heatmap(mapelites=mapelites,
                                          pop_name=pop_name,
                                          metric_name=metric_name,
                                          method_name=method_name)
    elif event_trig == 'reset-btn':
        res = _apply_reset(mapelites=mapelites,
                          logger=logger)
        if res:
            gen_counter = 0
            curr_heatmap = _build_heatmap(mapelites=mapelites,
                                          pop_name=pop_name,
                                          metric_name=metric_name,
                                          method_name=method_name)
    elif event_trig == 'b0-dropdown' or event_trig == 'b1-dropdown':
        res = _apply_bc_change(mapelites=mapelites,
                               logger=logger)
        if res:
            curr_heatmap = _build_heatmap(mapelites=mapelites,
                                          pop_name=pop_name,
                                          metric_name=metric_name,
                                          method_name=method_name)
    elif event_trig == 'subdivide-btn':
        res = _apply_bin_subdivision(mapelites=mapelites,
                                     selected_bins=selected_bins,
                                     logger=logger)
        if res:
            curr_heatmap = _build_heatmap(mapelites=mapelites,
                                          pop_name=pop_name,
                                          metric_name=metric_name,
                                          method_name=method_name)
            selected_bins = []
    elif event_trig == 'lsystem-modules':
        res = _apply_modules_update(mapelites=mapelites,
                                    modules=modules,
                                    logger=logger)
    elif event_trig == 'update-rules-btn':
        res = _apply_rules_update(mapelites=mapelites,
                                  rules=rules,
                                  logger=logger)
    # event_trig is a str of a dict, ie: '{"index":*,"type":"fitness-sldr"}', go figure
    elif event_trig is not None and 'fitness-sldr' in event_trig:
        res = _apply_fitness_reweight(mapelites=mapelites,
                                      weights=weights,
                                      logger=logger)
        if res:
            curr_heatmap = _build_heatmap(mapelites=mapelites,
                                     pop_name=pop_name,
                                     metric_name=metric_name,
                                     method_name=method_name)
    elif event_trig == 'population-dropdown' or event_trig == 'metric-dropdown' or event_trig == 'method-radio':
        curr_heatmap = _build_heatmap(mapelites=mapelites,
                                     pop_name=pop_name,
                                     metric_name=metric_name,
                                     method_name=method_name)
    elif event_trig == 'heatmap-plot' or event_trig == 'population_dropdown':
        i, j = _from_bc_to_idx(bcs=(clickData['points'][0]['x'],
                                    clickData['points'][0]['y']),
                               mapelites=mapelites)
        if mapelites.bins[j, i].non_empty(pop='feasible' if pop_name == 'Feasible' else 'infeasible'):
            curr_content = _get_elite_content(mapelites=mapelites,
                                              bin_idx=(j, i),
                                              pop='feasible' if pop_name == 'Feasible' else 'infeasible')
            if not mapelites.enforce_qnt and selected_bins != []:
                if [i, j] not in selected_bins:
                    selected_bins.append([i, j])
                else:
                    selected_bins.remove([i, j])
            else:
                selected_bins = [[i, j]]
            cs_string = cs_size = cs_n_blocks = ''
            if len(selected_bins) > 0:
                elite = mapelites.get_elite(bin_idx=_switch([selected_bins[-1]])[0],
                                            pop='feasible' if pop_name == 'Feasible' else 'infeasible')
                cs_string = elite.string
                cs_size = f'Spaceship size: {elite.size}'
                cs_n_blocks = f'Number of blocks: {elite.n_blocks}'
        else:
            logger.log(msg=f'Empty bin selected ({i}, {j}).')
    elif event_trig == 'selection-btn':
        _ = _apply_bin_selection_toggle(mapelites=mapelites,
                                        logger=logger)
        if mapelites.enforce_qnt and selected_bins:
            selected_bins = [selected_bins[-1]]
    elif event_trig == 'selection-clr-btn':
        logger.log(msg='Cleared bins selection.')
        selected_bins = []
        curr_content = go.Figure(data=[])
        cs_string = cs_size = cs_n_blocks  = ''
    elif event_trig == 'emitter-dropdown':
        _ = _apply_emitter_change(mapelites=mapelites,
                                  emitter_name=emitter_name,
                                  logger=logger)
    elif event_trig is None:
        curr_heatmap = _build_heatmap(mapelites=mapelites,
                                    pop_name=pop_name,
                                    metric_name=metric_name,
                                    method_name=method_name)
    else:
        logger.log(msg=f'Unrecognized event trigger: {event_trig}. No operations have been applied!')

    return curr_heatmap, curr_content, f'Valid bins are: {_get_valid_bins(mapelites=mapelites)}', f'Current generation: {gen_counter}', json.dumps(gen_counter), json_dumps(mapelites), str(mapelites.lsystem.hl_solver.parser.rules), f'Selected bin(s): {selected_bins}', json.dumps([[int(x[0]), int(x[1])] for x in selected_bins]), cs_string, cs_size, cs_n_blocks, json_dumps(obj=logger), gen_counter < N_GENS_ALLOWED, len(selected_bins) == 0
