import os
import sys

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    os.chdir(sys._MEIPASS)

from pcgsepy.evo.fitness import (Fitness, box_filling_fitness,
                                 func_blocks_fitness, mame_fitness,
                                 mami_fitness)
from pcgsepy.evo.genops import expander
from pcgsepy.guis.main_webapp.webapp import (app, set_app_layout,
                                             set_callback_props)
from pcgsepy.mapelites.behaviors import (BehaviorCharacterization, avg_ma,
                                         mame, mami, symmetry)
from pcgsepy.mapelites.buffer import Buffer, mean_merge
from pcgsepy.mapelites.emitters import *
from pcgsepy.mapelites.map import MAPElites
from pcgsepy.nn.estimators import MLPEstimator
from pcgsepy.setup_utils import get_default_lsystem, setup_matplotlib

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

mapelites = MAPElites(lsystem=lsystem,
                    feasible_fitnesses=feasible_fitnesses,
                    behavior_descriptors=(behavior_descriptors[0], behavior_descriptors[1]),
                    estimator=MLPEstimator(xshape=len(feasible_fitnesses),
                                            yshape=1),
                    buffer = Buffer(merge_method=mean_merge),
                    emitter=ContextualBanditEmitter(),
                    n_bins=(8, 8))
mapelites.emitter.sampling_strategy = 'thompson'

set_callback_props(mapelites=mapelites)

set_app_layout(behavior_descriptors_names=behavior_descriptors_names,
               
               mapelites=mapelites,
               
               dev_mode=True)
app.run_server(debug=False, host='127.0.0.1', port = 8080, use_reloader=False)
