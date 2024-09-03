import sys

import h5py
import numpy as np
from os.path import join, exists
import csv
import gzip

from config import (DATASET, POPSIZE, CONTROL_POINT, REPORT_NAME, MUTEXTENT, MUTATION_RECORD, NGEN,
                    ATTENTION, MUTATION_TYPE, MAX_ATTEMPT)
from individual import Individual
import vectorization_tools
from attention_manager import AttentionManager
from predictor import Predictor
from digit_mutator import DigitMutator
import time

def load_dataset(popsize):
    # Load the dataset.
    hf = h5py.File(DATASET, 'r')
    x_test = hf.get('xn')
    x_test = np.array(x_test)
    assert (x_test.shape[0] >= popsize)
    x_test = x_test[0:popsize]
    y_test = hf.get('yn')
    y_test = np.array(y_test)
    y_test = y_test[0:popsize]
    return x_test, y_test


def load_mnist_test(popsize, number):
    file_test_x = './original_dataset/t10k-images-idx3-ubyte.gz'
    file_test_y = './original_dataset/t10k-labels-idx1-ubyte.gz'

    with gzip.open(file_test_x, 'rb') as f:
        _ = np.frombuffer(f.read(16), dtype=np.uint8, count=4)
        images = np.frombuffer(f.read(), dtype=np.uint8)
        test_x = images.reshape(-1, 28, 28)

    with gzip.open(file_test_y, 'rb') as f:
        _ = np.frombuffer(f.read(8), dtype=np.uint8, count=2)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        test_y = labels

    idx = [i for i, label in enumerate(test_y) if label == number]
    #print(f"number of {number} is {len(idx)}")
    filtered_test_y = test_y[idx]
    filtered_test_x = test_x[idx]

    if popsize < filtered_test_y.shape[0]:
        select_index = np.random.choice(range(filtered_test_x.shape[0]), size=popsize, replace=False)
        select_index = np.sort(select_index)
        # print(f"select index {select_index}")
        return filtered_test_x[select_index], filtered_test_y[select_index]
    else:
        return filtered_test_x, filtered_test_y


def generate_digit(ind_id, image, label):
    xml_desc = vectorization_tools.vectorize(image)
    return Individual(ind_id, xml_desc, label)


class Population:
    def __init__(self, number, pop_size=POPSIZE, xai_method=ATTENTION,
                 enable_timer: bool = False):
        #x_test, y_test = load_dataset(pop_size)
        self.digit = number

        x_test, y_test = load_mnist_test(pop_size, self.digit)
        self.population_to_mutate = [generate_digit(i, image, label) for i, (image, label) in
                                     enumerate(zip(x_test, y_test))]
        self.size = len(self.population_to_mutate)
        self.misclass_number = 0
        self.misclass_list = []
        self.finished_population = []
        self.xai_method = xai_method
        if CONTROL_POINT == "random":
            self.xai_method = "none"
        self.attention = AttentionManager(num=self.digit, attention_method=self.xai_method)

        self.enable_timer = enable_timer
        if self.enable_timer:
            self.elapsed_time = {
                "mutation_time": 0,
                "mutation_number": 0,
                "evaluation_number": 0,
                "evaluation_time": 0, # also includes heatmap time
                "heatmap_time": 0
            }
    def evaluate_population(self, gen_number, folder):
        # batch evaluation for
        #         Individual.predicted_label
        #         Individual.confidence
        #         Individual.misclass
        #         Individual.attention
        #         Population.misclass_number

        #         Individual.fail
        # in initialization or after every mutation of all population

        # Prediction
        if self.enable_timer:
            start_evaluate = time.time()
            self.elapsed_time["evaluation_number"] += len(self.population_to_mutate)

        batch_individual = [ind.purified for ind in self.population_to_mutate]
        batch_individual = np.reshape(batch_individual, (-1, 28, 28, 1))
        batch_label = ([ind.expected_label for ind in self.population_to_mutate])

        if CONTROL_POINT != "random":
            if self.enable_timer:
                start_hm = time.time()
            attmaps = self.attention.compute_attention_maps(batch_individual)
            attmaps = np.reshape(attmaps, (-1, 28, 28))
            if self.enable_timer:
                self.elapsed_time["heatmap_time"] += time.time() - start_hm
        else:
            attmaps = [None] * len(batch_individual)

        predictions, confidences = (Predictor.predict(img=batch_individual,
                                                      label=batch_label))
        # label result and detect misclass
        for ind, prediction, confidence, attmap \
                in zip(self.population_to_mutate, predictions, confidences, attmaps):
            ind.confidence = confidence
            ind.predicted_label = prediction

            ind.misclass = ind.expected_label != ind.predicted_label

            ind.attention = attmap

            if MUTATION_RECORD:
                ind.append_mutation_log(gen_number, folder)

        for ind in self.population_to_mutate:
            if ind.misclass or ind.fail:
                self.population_to_mutate.remove(ind)
                ind.mutate_attempts = gen_number
                self.finished_population.append(ind)
                if ind.misclass:
                    self.misclass_number += 1

                if MUTATION_RECORD:
                    ind.save_mutation_log(folder)
        if gen_number >= NGEN and MUTATION_RECORD:
            for ind in self.population_to_mutate:
                ind.save_mutation_log(folder)

        if self.enable_timer:
            self.elapsed_time["evaluation_time"] += time.time() - start_evaluate

    def mutate(self):
        if self.enable_timer:
            start_mutation = time.time()
        batch_individual = [ind for ind in self.population_to_mutate]
        for ind in batch_individual:
            mutator = DigitMutator(ind)
            mutator.control_points_select(CONTROL_POINT)
            attempt = 0
            while True:
                try:
                    attempt += 1
                    mutator.mutate()
                    break
                except Exception:
                    print(f"Retry mutation, attempts {attempt}.")
                    if attempt >= MAX_ATTEMPT:
                        print(f"Mutation failed!")
                        #sys.exit()
                        ind.fail = True
                        break

        if self.enable_timer:
            self.elapsed_time["mutation_time"] += time.time() - start_mutation
            self.elapsed_time["mutation_number"] += len(batch_individual)

    def create_report(self, path, gen_number):
        dst = join(path, REPORT_NAME)
        with open(dst, mode='w') as report_file:
            report_writer = csv.writer(report_file,
                                       delimiter=',',
                                       quotechar='"',
                                       quoting=csv.QUOTE_MINIMAL)
            report_writer.writerow(['population size',
                                    'total iteration number',
                                    'misbehaviour number',
                                    'mutation type',
                                    'stride extent'])
            report_writer.writerow([self.size,
                                    gen_number,
                                    self.misclass_number,
                                    CONTROL_POINT,
                                    MUTEXTENT])

            report_writer.writerow('')

            report_writer.writerow(['id',
                                    'expected_label',
                                    'predicted_label',
                                    'misbehaviour',
                                    'confidence',
                                    'mutate_attempts',
                                    'failed'])

            for ind in [*self.finished_population, *self.population_to_mutate]:
                report_writer.writerow([ind.id,
                                        ind.expected_label,
                                        ind.predicted_label,
                                        ind.misclass,
                                        ind.confidence,
                                        ind.mutate_attempts,
                                        ind.fail])

        # summary table
        table_name = './runs/summary.csv'


        data = [self.size,
                self.digit,
                gen_number,
                self.xai_method,
                CONTROL_POINT,
                MUTATION_TYPE,
                MUTEXTENT,
                self.misclass_number,
                "{:.2f}%".format(self.misclass_number/self.size*100)]

        # create csv if it's not existed
        if not exists(table_name):
            with open(table_name, 'w', newline='') as file:
                writer = csv.writer(file)
                headers = ['population size',
                           'label number',
                           'total iteration number',
                           'attention method',
                           'control point',
                           'mutation type',
                           'stride extent',
                           'misbehaviour number',
                           'misbehaviour Percentage']
                writer.writerow(headers)
                writer.writerow(data)
        else:
            # only write data when csv is already existed
            with open(table_name, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)


if __name__ == "__main__":
    from utils import get_distance
    from folder import Folder

    pop_size = 3

    pop = Population(number=5, pop_size=pop_size)
    print(f" Population size {pop.size}")
    # pop.evaluate_population(0)

    for idx in range(2):
        pop.evaluate_population(idx, Folder)
        # print([ind.confidence for ind in pop.population_to_mutate])
        pop.mutate()

        # print([get_distance(digit_ini[i], digit_cur[i]) for i in range(POPSIZE)])

    print(f" Misclass number {pop.misclass_number}")
