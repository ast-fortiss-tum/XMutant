import time

import numpy as np
import random
import csv

from config import POPSIZE, NGEN, XAI_METHOD, SEED
from population import Population
# from timer import Timer
import time
from folder import Folder
from utils import set_all_seeds

# import tensorflow as tf
# print(tf.test.is_gpu_available())
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

set_all_seeds(SEED)

def main(popsize=POPSIZE, xai_method=XAI_METHOD):

    # folder = Folder(num=num, xai_method=xai_method)
    pop = Population(pop_size=popsize, xai_method=xai_method)
    print(f" Population size {pop.size}")

    folder = Folder(xai_method=xai_method)
    condition = True
    gen = 1

    while condition:
        pop.evaluate_population(gen, folder)
        confidences = [ind.confidence for ind in pop.population_to_mutate]
        if len(confidences) > 0:
            print('Iteration:{:4}, Mis-number:{:3}, Pop-number:{:3}, avg:{:1.10f}, min:{:2.4f}, max:{:1.4f}'
                  .format(*[gen, pop.misclassified_number, len(confidences), np.mean(confidences),
                            np.min(confidences), np.max(confidences)]))
            pop.mutate_population()
            gen += 1

            if gen == NGEN:
                condition = False

        else:
            print("All mutations finished, early termination")
            condition = False
    if len(pop.population_to_mutate) > 0:
        pop.evaluate_population(gen, folder)
        """confidences = [ind.confidence for ind in pop.population_to_mutate]
        print('Iteration:{:4}, Mis-number:{:3}, Pop-number:{:3}, avg:{:1.10f}, min:{:2.4f}, max:{:1.4f}'
              .format(*[gen, pop.misclass_number, len(confidences), np.mean(confidences),
                        np.min(confidences), np.max(confidences), ]))"""

    print(f"MUTATION FINISHED")
    # record data
    pop.create_report(folder.DST, gen, SEED)
    print("REPORT GENERATED")



if __name__ == "__main__":
    main(popsize=3)
    
    # random test
    # for digit in range(10): # range(10):
    #     set_all_seeds(digit)
    #     main(num=digit, popsize=1, xai_method="SmoothGrad", enable_timer=True)

    # for xai_method in ["SmoothGrad", "GradCAM++", "Faster-ScoreCAM", "IntegratedGradients"]:
    #     for digit in range(10):
    #         set_all_seeds(digit)
    #         main(num=digit, popsize=10, xai_method=xai_method, enable_timer=True)
