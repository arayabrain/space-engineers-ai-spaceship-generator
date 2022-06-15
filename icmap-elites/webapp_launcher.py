import argparse
import datetime
from secrets import choice

from pcgsepy.common.jsonifier import json_dumps, json_loads
from pcgsepy.evo.fitness import (Fitness, bounding_box_fitness,
                                 box_filling_fitness, func_blocks_fitness,
                                 mame_fitness, mami_fitness)
from pcgsepy.evo.genops import expander
from pcgsepy.fi2pop.utils import MLPEstimator
from pcgsepy.guis.main_webapp.webapp import app, set_app_layout, set_callback_props
from pcgsepy.mapelites.behaviors import (BehaviorCharacterization, avg_ma,
                                         mame, mami, symmetry)
from pcgsepy.mapelites.buffer import Buffer, max_merge, mean_merge, min_merge
from pcgsepy.mapelites.emitters import *
from pcgsepy.mapelites.map import MAPElites
from pcgsepy.setup_utils import get_default_lsystem, setup_matplotlib

parser = argparse.ArgumentParser()
parser.add_argument("--mapelites_file", help="Location of the MAP-Elites object",
                    type=str, default=None)
parser.add_argument("--dev_mode", help="Launch the webapp in developer mode",
                    action='store_true')
parser.add_argument("--debug", help="Launch the webapp in debug mode",
                    action='store_true')
parser.add_argument("--emitter", help="Specify the emitter type",
                    type=str, choices=['random', 'preference-matrix', 'contextual-bandit'], default='random')
parser.add_argument("--host", help="Specify host address",
                    type=str, default='127.0.0.1')
parser.add_argument("--port", help="Specify port",
                    type=int, default=8050)
parser.add_argument("--use_reloader", help="Use reloader (set to True when using Jupyter Notebooks)",
                    action='store_false')

args = parser.parse_args()

setup_matplotlib(larger_fonts=False)

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

feasible_fitnesses = [
    #     Fitness(name='BoundingBox',
    #             f=bounding_box_fitness,
    #             bounds=(0, 1)),
    Fitness(name='BoxFilling',
            f=box_filling_fitness,
            bounds=(0, 1)),
    Fitness(name='FuncionalBlocks',
            f=func_blocks_fitness,
            bounds=(0, 1)),
    Fitness(name='MajorMediumProportions',
            f=mame_fitness,
            bounds=(0, 1)),
    Fitness(name='MajorMinimumProportions',
            f=mami_fitness,
            bounds=(0, 1))
]

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

behavior_descriptors_names = [x.name for x in behavior_descriptors]

if args.mapelites_file:
    with open(args.mapelites_file, 'r') as f:
        mapelites = json_loads(f.read())
else:
    emitter_choices = {
		'random': RandomEmitter(),
		'preference-matrix': HumanPrefMatrixEmitter(),
  		'contextual-bandit': ContextualBanditEmitter()
	}
    mapelites = MAPElites(lsystem=lsystem,
                          feasible_fitnesses=feasible_fitnesses,
                          behavior_descriptors=(behavior_descriptors[0], behavior_descriptors[1]),
                          estimator=MLPEstimator(xshape=len(feasible_fitnesses),
                                                 yshape=1),
                          buffer = Buffer(merge_method=mean_merge),
                          emitter=emitter_choices[args.emitter],
                          n_bins=(8, 8))
    mapelites.emitter.diversity_weight = 0.25
    mapelites.generate_initial_populations()
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f'{t}_mapelites_{mapelites.emitter.name}_gen00', 'w') as f:
        f.write(json_dumps(mapelites))

set_callback_props(mapelites=mapelites)

set_app_layout(mapelites=mapelites,
               behavior_descriptors_names=behavior_descriptors_names,
               dev_mode=args.dev_mode)

app.run_server(debug=args.debug,
               host=args.host,
               port=args.port,
               use_reloader=args.use_reloader)
