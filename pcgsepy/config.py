import configparser
import os

config = configparser.ConfigParser()
curr_dir = os.getcwd()
config.read(os.path.join(curr_dir, 'configs.ini'))

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
N_APE = config['L-SYSTEM'].getint('n_axioms_generated')

# lower bound for mutation
MUTATION_LOW = config['GENOPS'].getint('mutations_lower_bound')
# upper bound for mutation
MUTATION_HIGH = config['GENOPS'].getint('mutations_upper_bound')
# initial mutation probability
MUTATION_INITIAL_P = config['GENOPS'].getfloat('mutations_initial_p')
# mutation decay
MUTATION_DECAY = config['GENOPS'].getfloat('mutations_decay')
# crossover probability
CROSSOVER_P = config['GENOPS'].getfloat('crossover_p')

POP_SIZE = config['FI2POP'].getint('population_size')
N_RETRIES = config['FI2POP'].getint('n_initial_retries')
N_GENS = config['FI2POP'].getint('n_generations')

USE_BBOX = config['FITNESS'].get('use_bounding_box')
if USE_BBOX:
    bbox = config['FITNESS'].get('bounding_box').split(',')
    BBOX_X, BBOX_Y, BBOX_Z = float(bbox[0]), float(bbox[1]), float(bbox[2])
MAME_MEAN = config['FITNESS'].getfloat('mame_mean')
MAME_STD = config['FITNESS'].getfloat('mame_std')
MAMI_MEAN = config['FITNESS'].getfloat('mami_mean')
MAMI_STD = config['FITNESS'].getfloat('mami_std')
FUTO_MEAN = config['FITNESS'].getfloat('futo_mean')
FUTO_STD = config['FITNESS'].getfloat('futo_std')
TOVO_MEAN = config['FITNESS'].getfloat('tovo_mean')
TOVO_STD = config['FITNESS'].getfloat('tovo_std')

BIN_POP_SIZE = config['MAPELITES'].getint('bin_population')
CS_MAX_AGE = config['MAPELITES'].getint('max_age')

N_RUNS = config['EXPERIMENT'].getint('n_runs')
EXP_NAME = config['EXPERIMENT'].get('exp_name')