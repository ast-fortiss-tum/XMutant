# Make sure that any of this properties can be overridden using env.properties
from os.path import join
import json

# GA Setup
POPSIZE = 1000
NGEN = 100
SEED = 44
#
# RUNTIME = 3600
#
MODEL = "models/imdb_lstm_model.h5"

XAI_METHOD = "Lime"
# "Random"|"SmoothGrad"|"VanillaSaliency"|"Lime"|"IntegratedGradients"

# Control point selection methods

# Mutation Hyperparameters
# MUTATION_TYPE =
MAX_ATTEMPT = 5
REPORT_NAME = 'stats.csv'
MUTATION_RECORD = True
LENGTH_EXPLANATION = 20
#
# FEATURES = ["PosCount", "NegCount"] #PosCount NegCount VerbCount
# NUM_CELLS = 25
# RUN = 1
# NAME = f"RUN_{RUN}_{POPSIZE}_{FEATURES[0]}-{FEATURES[1]}_{RUNTIME}"
#
# EXPECTED_LABEL = 1 # 0 or 1
# MUTLOWERBOUND = 0.01
# MUTUPPERBOUND = 0.6
#
# SELECTIONOP = 'ranked' # random or ranked or dynamic_ranked
# SELECTIONPROB = 0.5
# RANK_BIAS = 1.5 # value between 1 and 2
# RANK_BASE = 'contribution_score' # perf or density or contribution_score
#
# INITIAL_POP = 'seeded'
#
# ORIGINAL_SEEDS = "starting_seeds_pos.txt"

# Data configuration
MAX_SEQUENCE_LENGTH = 700 #  600 (95%) 700 (97%)  #100
NUM_DISTINCT_WORDS = 17000 # 10654 (95%) 16909 (97%) 34664 (99%)
EMBEDDING_OUTPUT_DIMS = 128 #15

INDEX_FROM = 3
DEFAULT_WORD_ID = {
    "<pad>": 0,
    "<start>": 1,
    "<unk>": 2,
    "<unused>": 3
}

# Model configuration
LSTM_CONFIG = {

}
loss_function = 'binary_crossentropy'
optimizer = 'adam'
additional_metrics = ['accuracy']
number_of_epochs = 100
verbosity_mode = True
validation_split = 0.20


def to_json(folder):
    config = {
        'popsize': str(POPSIZE),
        'model': str(MODEL),
        'runtime': str(RUNTIME),
        'features': str(FEATURES),
        'mut low': str(MUTLOWERBOUND),
        'mut up': str(MUTUPPERBOUND),
        'ranked prob': str(SELECTIONPROB),
        'rank bias': str(RANK_BIAS),
        'rank base': str(RANK_BASE),
        'selection': str(SELECTIONOP),
        'expected label': str(EXPECTED_LABEL)
    }
    filedest = join(folder, "config.json")
    with open(filedest, 'w') as f:
        (json.dump(config, f, sort_keys=True, indent=4))
