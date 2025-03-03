import time

import numpy as np
import random
import csv

from config import DATASET, POPSIZE, STOP_CONDITION, NGEN, NUMBER, ATTENTION
from population import Population
from timer import Timer
import time
from folder import Folder
from utils import set_all_seeds

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(num=NUMBER, popsize=POPSIZE, xai_method=ATTENTION, enable_timer: bool = False):
    if enable_timer:
        start_time = time.time()

    folder = Folder(num=num, xai_method=xai_method)
    pop = Population(number=num, pop_size=popsize, xai_method=xai_method, enable_timer=enable_timer)
    print(f" Population size {pop.size}")

    # Collect data
    # field_names = ["id", "misclass ", "predicted label"]
    # data = []

    condition = True
    gen = 1

    while condition:
        pop.evaluate_population(gen, folder)
        confidences = [ind.confidence for ind in pop.population_to_mutate]
        if len(confidences) > 0:
            print('Iteration:{:4}, Mis-number:{:3}, Pop-number:{:3}, avg:{:1.10f}, min:{:2.4f}, max:{:1.4f}'
                  .format(*[gen, pop.misclass_number, len(confidences), np.mean(confidences),
                            np.min(confidences), np.max(confidences)]))
            pop.mutate()
            gen += 1
            if STOP_CONDITION == "iter":
                if gen == NGEN:
                    condition = False
            elif STOP_CONDITION == "time":
                if not Timer.has_budget():
                    condition = False
        else:
            print("All mutations finished, early termination")
            condition = False
    if len(pop.population_to_mutate) > 0:
        pop.evaluate_population(gen, folder)
        # confidences = [ind.confidence for ind in pop.population_to_mutate]
        """print('Iteration:{:4}, Mis-number:{:3}, Pop-number:{:3}, avg:{:1.10f}, min:{:2.4f}, max:{:1.4f}'
              .format(*[gen, pop.misclass_number, len(confidences), np.mean(confidences),
                        np.min(confidences), np.max(confidences), ]))"""

    print(f"MUTATION digit{num} FINISHED")
    # record data
    pop.create_report(folder.DST, gen)
    print("REPORT GENERATED")

    if enable_timer:
        print("TIMING REPORT GENERATED")
        print(f"digit {num}, popsize {popsize}, xai_method {xai_method}")
        total_elapsed_time = time.time() - start_time
        print("Elapsed time:", total_elapsed_time)
        print(pop.elapsed_time)
        print(f"percentage of heatmap time {pop.elapsed_time['heatmap_time'] / (total_elapsed_time)}")
        return pop.elapsed_time, total_elapsed_time


if __name__ == "__main__":
    digit = 0
    set_all_seeds(digit)
    main(num=digit, popsize=200)

    """for xai_method in ["SmoothGrad", "GradCAM++", "Faster-ScoreCAM", "IntegratedGradients"]:
        for digit in range(0, 2):# range(10):
            set_all_seeds(digit)
            main(num=digit, popsize=POPSIZE, xai_method=xai_method)"""

    # # random vit_model
    # for digit in range(10): # range(10):
    #     set_all_seeds(digit)
    #     main(num=digit, popsize=POPSIZE, xai_method="SELF", enable_timer=True)

    # for xai_method in ["SmoothGrad", "GradCAM++", "Faster-ScoreCAM", "IntegratedGradients"]:
    #     for digit in range(10):
    #         set_all_seeds(digit)
    #         main(num=digit, popsize=10, xai_method=xai_method, enable_timer=True)