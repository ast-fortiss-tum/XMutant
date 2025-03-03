DJ_DEBUG = 1

# GA Setup
POPSIZE = 200

num_classes = 10

INITIALPOP = 'seeded'

GENERATE_ONE_ONLY = False

MODEL = 'models/cnnClassifier.h5'
# MODEL = 'models/cnn-classifier-low.h5'

RESULTS_PATH = 'results'
REPORT_NAME = 'stats.csv'
DATASET = 'original_dataset/janus_dataset_comparison.h5'



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
SEEDS_LIST_FOR_REPETITIONS_OF_MUTATIONS = [4398, 980, 987423, 99982, 1123, 4098, 1946, 22601, 55037, 812109, 53898,
                                           187988]

