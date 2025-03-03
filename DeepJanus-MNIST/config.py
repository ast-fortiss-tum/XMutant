import os
DJ_DEBUG = 1

# GA Setup
POPSIZE = 100

DATALOADER = 'xmutant' # 'normal' or 'xmutant'
DATASET_NAME = 'MNIST'  #  'MNIST', 'FashionMNIST',

STOP_CONDITION = "iter"
#STOP_CONDITION = "time"



SEED = 0

NGEN = 200
RUNTIME = 3600
STEPSIZE = 10
# Mutation Hyperparameters
# range of the mutation
MUTLOWERBOUND = 0.01
MUTUPPERBOUND = 0.6
MUTEXTENT = 1


EXPLABEL = int(os.getenv('EXPECT_LABEL', default = 5))

ARCHIVE_THRESHOLD = float(os.getenv('AR_THRES', default = 1)) # 1 #0.1#1 #4.0
# Attention map method
attention_method = os.getenv('XAI', default = "SmoothGrad") #
#"SmoothGrad" #"GradCAM++"#"SmoothGrad" # None
# "SmoothGrad"|"VanillaSaliency"|# "GradCAM"|"GradCAM++"|"ScoreCAM"|"Faster-ScoreCAM"|"IntegratedGradients"
# "SmoothGrad"|"GradCAM++"|"Faster-ScoreCAM"|"IntegratedGradients"
# Control point selection methods
control_point =  os.getenv('SELECTION', default = "clustering" )    # |"random"|"square-window"|"clustering"
# Mutation Hyperparameters
mutation_type = os.getenv('DIRECTION', default = "random_cycle" )    # |"random"|"random_cycle"|"toward_centroid"|"backward_centroid"|"centroid_based"

XMUTANT_CONFIG = {"xai":attention_method, "selection": control_point, "direction": mutation_type}
if attention_method == "none" or control_point == "random":
    XMUTANT_CONFIG["xai"] = None
    XMUTANT_CONFIG["selection"] = "random"
    XMUTANT_CONFIG["direction"] = "random"

# Reseeding Hyperparameters
# extent of the reseeding operator
RESEEDUPPERBOUND = 10

K_SD = 0.1

# K-nearest
K = 1



#------- NOT TUNING ----------

# mutation operator probability
MUTOPPROB = 0.5
MUTOFPROB = 0.5

IMG_SIZE = 28
num_classes = 10

INITIALPOP = 'seeded'

GENERATE_ONE_ONLY = True

MODEL2 = 'models/cnnClassifier_lowLR.h5'
MODEL = 'models/cnnClassifier.h5'
#MODEL = "models/regular3"
#MODEL = 'models/cnnClassifier_001.h5'
#MODEL = 'models/cnnClassifier_op.h5'

# RESULTS_PATH = 'results'
REPORT_NAME = 'stats.csv'
DATASET = 'original_dataset/janus_dataset_comparison.h5'

#TODO: set interpreter
INTERPRETER = '/home/vin/yes/envs/tf_gpu/bin/python'