DJ_DEBUG = 1

# GA Setup
POPSIZE = 200

STOP_CONDITION = "iter"
# STOP_CONDITION = "time"

NGEN = 100
RUNTIME = 3600
STEPSIZE = 10

# Ground Truth Number
NUMBER = 5

# Attention map method
ATTENTION = "SmoothGrad"
# "SmoothGrad"|"VanillaSaliency"|# "GradCAM"|"GradCAM++"|"ScoreCAM"|"Faster-ScoreCAM"|"IntegratedGradients"
# "SmoothGrad"|"GradCAM++"|"Faster-ScoreCAM"|"IntegratedGradients"

# Control point selection methods
CONTROL_POINT = "clustering"  # |"random"|"square-window"|"clustering"
# Mutation Hyperparameters
MUTATION_TYPE = "toward_centroid"  # |"random"|"random_cycle"|"toward_centroid"|"backward_centroid"|"centroid_based"

# Max mutation retry times
MAX_ATTEMPT = 15

# range of the mutation
MUTLOWERBOUND = 0.01
MUTUPPERBOUND = 0.6
MUTEXTENT = 2
# Reseeding Hyperparameters
# extent of the reseeding operator
RESEEDUPPERBOUND = 10


# K-nearest
K = 1

# Mutation Recording
MUTATION_RECORD = True

# ------- NOT TUNING ----------

# mutation operator probability
MUTOPPROB = 0.5
MUTOFPROB = 0.5

IMG_SIZE = 28
num_classes = 10

INITIALPOP = "seeded"

GENERATE_ONE_ONLY = False

MODEL = "models/cnnClassifier.h5"
# MODEL = 'models/cnn-classifier-low.h5'

RESULTS_PATH = "results"
REPORT_NAME = "stats.csv"
DATASET = "original_dataset/janus_dataset_comparison.h5"


SAVE_IMAGES = True
START_INDEX_DATASET = 0
EXTENT = 0.2
EXTENT_STEP = 0.1
EXTENT_LOWERBOUND = 0.01
EXTENT_UPPERBOUND = 0.6
# NUMBER_OF_POINTS = 6
SQUARE_SIZE = 3
NUMBER_OF_MUTATIONS = 10
NUMBER_OF_REPETITIONS = 5
SEEDS_LIST_FOR_REPETITIONS_OF_MUTATIONS = [
    4398,
    980,
    987423,
    99982,
    1123,
    4098,
    1946,
    22601,
    55037,
    812109,
    53898,
    187988,
]
