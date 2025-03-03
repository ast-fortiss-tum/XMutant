
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
ATTENTION = "SELF"
# "SmoothGrad"|"GradCAM++"|"Faster-ScoreCAM"|"IntegratedGradients"

# Control point selection methods
CONTROL_POINT = "random"   # |"random"|"square-window"|"clustering"
# Mutation Hyperparameters
MUTATION_TYPE = "random_cycle"  # |"random"|"random_cycle"|"toward_centroid"|"backward_centroid"|"centroid_based"

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

INITIALPOP = 'seeded'

GENERATE_ONE_ONLY = False


# VIT_MODEL_CONFIGS = {
#     'image_size': 32,
#     'channel_size': 1,
#     'patch_size': 4,
#     'embed_size': 512,
#     'num_heads': 8,
#     'classes': 10,
#     'num_layers': 3,
#     'hidden_size': 256,
#     'dropout': 0.2,
#     'checkpoint_path': 'transformer_model/checkpoints/epoch_23_val_acc_98.67.pth'
# }

VIT_MODEL_CONFIGS = {
    'image_size': 28,
    'embed_dim': 256,
    'num_heads': 8,
    'num_layers': 6,
    'patch_size': 4,
    'num_channels': 1,
    'num_classes': 10,
    'dropout': 0.2,
    'checkpoint_path': 'vit_model/checkpoints/4_49_patches_epoch_24_val_acc_97.53.pth',
    'normalization': True
}
VIT_MODEL_CONFIGS['hidden_dim'] = VIT_MODEL_CONFIGS['embed_dim'] * 3
VIT_MODEL_CONFIGS['num_patches'] = (VIT_MODEL_CONFIGS['image_size'] // VIT_MODEL_CONFIGS['patch_size']) ** 2
BATCH_SIZE = 64

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

