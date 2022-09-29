import configparser
import os

config = configparser.ConfigParser()
curr_dir = os.getcwd()
config.read(os.path.join(curr_dir, 'configs.ini'))

USE_TORCH = config['LIBRARY'].getboolean('use_torch')

HOST = config['API'].get('host')
PORT = config['API'].getint('port')

# json file with common atoms and action+args
COMMON_ATOMS = config['L-SYSTEM'].get('common_atoms')
# json file with high level atoms and dimensions
HL_ATOMS = config['L-SYSTEM'].get('hl_atoms')
# ranges of parametric l-system
pl_range = config['L-SYSTEM'].get('pl_range').strip().split(',')
PL_LOW, PL_HIGH = int(pl_range[0]), int(pl_range[1])
# required tiles for constraint components_constraint
REQ_TILES = config['L-SYSTEM'].get('req_tiles').split(',')
# L-system variables
# number of iterations (high level)
N_ITERATIONS = config['L-SYSTEM'].getint('n_iterations')
# number of axioms generated at each expansion step
N_SPE = config['L-SYSTEM'].getint('n_axioms_generated')

# initial mutation probability
MUTATION_INITIAL_P = config['GENOPS'].getfloat('mutations_initial_p')
# mutation decay
MUTATION_DECAY = config['GENOPS'].getfloat('mutations_decay')
# crossover probability
CROSSOVER_P = config['GENOPS'].getfloat('crossover_p')

# population size
POP_SIZE = config['FI2POP'].getint('population_size')
# number of initialization retries
N_RETRIES = config['FI2POP'].getint('n_initial_retries')
# number of generations
N_GENS = config['FI2POP'].getint('n_generations')
# maximum string length (-1 for unlimited lenght)
MAX_STRING_LEN = config['FI2POP'].getint('max_string_len')
# maximum patience when generating new pools
GEN_PATIENCE = config['FI2POP'].getint('gen_patience')

# use or don't use the bounding box fitness
USE_BBOX = config['FITNESS'].get('use_bounding_box')
if USE_BBOX:
    bbox = config['FITNESS'].get('bounding_box').split(',')
    # bounding box upper limits
    BBOX_X, BBOX_Y, BBOX_Z = float(bbox[0]), float(bbox[1]), float(bbox[2])
MAME_MEAN = config['FITNESS'].getfloat('mame_mean')
MAME_STD = config['FITNESS'].getfloat('mame_std')
MAMI_MEAN = config['FITNESS'].getfloat('mami_mean')
MAMI_STD = config['FITNESS'].getfloat('mami_std')
MAX_X_SIZE = config['MAPELITES'].getint('max_x_size')
MAX_Y_SIZE = config['MAPELITES'].getint('max_y_size')
MAX_Z_SIZE = config['MAPELITES'].getint('max_z_size')

# number of solutions per bin
BIN_POP_SIZE = config['MAPELITES'].getint('bin_population')
BIN_N = config['MAPELITES'].get('bin_n').split(',')
BIN_N = tuple([int(x) for x in BIN_N])
# maximum age of solutions
CS_MAX_AGE = config['MAPELITES'].getint('max_age')
# PCA dimensions
N_DIM_RED = config['MAPELITES'].getint('n_dimensions_reduced')
# maximum number of dimensions PCA can analyze
MAX_DIMS_RED = config['MAPELITES'].getint('max_possible_dimensions')
# minimum fitness assignable
EPSILON_F = config['MAPELITES'].getfloat('epsilon_fitness')
# interval for realigning infeasible fitnesses
ALIGNMENT_INTERVAL = config['MAPELITES'].getint('alignment_interval')
# rescale infeasible fitness with reporduction probability
RESCALE_INFEAS_FITNESS = config['MAPELITES'].getboolean('rescale_infeas_fitness')
# minimum subdivision percentage of a bin
BIN_SMALLEST_PERC = config['MAPELITES'].getfloat('bin_min_resolution')
# whether to use a linear estimator or a NN estimator in the emitter (if possible)
USE_LINEAR_ESTIMATOR = config['MAPELITES'].getboolean('use_linear_estimator')

# number of experiments to run
N_RUNS = config['EXPERIMENT'].getint('n_runs')
# name of the current experiment
EXP_NAME = config['EXPERIMENT'].get('exp_name')

# list of emitters
MY_EMITTERS = [f'{x}.json' for x in config['USER-STUDY'].get('emitters_list').split(',')]
# number of generations allowed per experiment
N_GENS_ALLOWED = config['USER-STUDY'].getint('n_generations_allowed')
# number of automated emitters steps
N_EMITTER_STEPS = config['USER-STUDY'].getint('n_emitter_steps')

CONTEXT_IDXS = config['USER-STUDY'].get('context_idxs').split(',')
CONTEXT_IDXS = [int(x) for x in CONTEXT_IDXS]

BETA_A = config['USER-STUDY'].getint('beta_a')
BETA_B = config['USER-STUDY'].getint('beta_b')
