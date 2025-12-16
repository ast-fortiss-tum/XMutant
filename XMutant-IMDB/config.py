# Setup
POPSIZE = 200
NGEN = 100
SEED = 44

# RUNTIME = 3600
#
MODEL = "models/imdb_lstm_model.h5"

XAI_METHOD = "SmoothGrad"
# "Random"|"SmoothGrad"|"VanillaSaliency"|"Lime"|"IntegratedGradients"

# Control point selection methods

# Mutation Hyperparameters
MAX_ATTEMPT = 5
REPORT_NAME = "stats.csv"
MUTATION_RECORD = True
LENGTH_EXPLANATION = 32

API_KEY = None  #  ChatGPT API key


# Data configuration
MAX_SEQUENCE_LENGTH = 700  #  600 (95%) 700 (97%)  #100
NUM_DISTINCT_WORDS = 17000  # 10654 (95%) 16909 (97%) 34664 (99%)
EMBEDDING_OUTPUT_DIMS = 128  # 15

INDEX_FROM = 3
DEFAULT_WORD_ID = {"<pad>": 0, "<start>": 1, "<unk>": 2, "<unused>": 3}

# Model configuration

loss_function = "binary_crossentropy"
optimizer = "adam"
additional_metrics = ["accuracy"]
number_of_epochs = 100
verbosity_mode = True
validation_split = 0.20
