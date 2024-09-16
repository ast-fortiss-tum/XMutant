
import numpy as np
from os.path import join, exists
import csv

from tensorflow.keras.datasets import imdb
from config import (POPSIZE, REPORT_NAME, MUTATION_RECORD, XAI_METHOD)
from config import (MAX_SEQUENCE_LENGTH, NUM_DISTINCT_WORDS,
                    DEFAULT_WORD_ID, INDEX_FROM)
from individual import Individual
from xai_imdb import xai_embedding, lime_batch_explainer
import utils
from predictor import Predictor
from mutation_manager import mutate, mutate_lime
import time
import logging as log

predictor = Predictor()

def load_imdb_test(pop_size, seed=0):
    (_, _), (x_test, y_test) = imdb.load_data(num_words=NUM_DISTINCT_WORDS,
                                                          start_char=DEFAULT_WORD_ID['<start>'],
                                                          oov_char=DEFAULT_WORD_ID['<unk>'],
                                                          index_from=INDEX_FROM)
    np.random.seed(seed=seed)
    assert pop_size < x_test.shape[0], "popsize must be smaller than training size"

    select_index = np.random.choice(range(x_test.shape[0]), size=pop_size, replace=False)
    select_index = np.sort(select_index)

    x_test_selected = x_test[select_index]
    y_test_selected = y_test[select_index]
    # check if label persevered
    y_prediction, _ = predictor.predict(x_test_selected)

    x_test_selected = x_test_selected[y_prediction == y_test_selected]
    y_test_selected = y_test_selected[y_prediction == y_test_selected]

    # Note: x_test is not padded, and pop size might be smaller than required due to not persevered labels

    return x_test_selected, y_test_selected


# x, y = load_imdb_test(popsize= 10)
# y_pre, _ = predictor.predict(x)

class Population:
    def __init__(self, pop_size=POPSIZE, xai_method=XAI_METHOD):

        x_test, y_test = load_imdb_test(pop_size)
        self.population_to_mutate = [Individual(id=i, token_ids=x, label=y) for i, (x, y) in
                                     enumerate(zip(x_test, y_test))]
        self.size = len(self.population_to_mutate)
        self.xai_method = xai_method

        self.misclassified_number = 0
        self.misclassified_list = []
        self.finished_population = []

        # self.attention = AttentionManager(num=self.digit, attention_method=self.xai_method)

    def evaluate_population(self, gen_number, folder): #
        # batch evaluation for
        #         Individual.predicted_label
        #         Individual.confidence
        #         Individual.misclassified
        #         Individual.attention
        #         Population.misclassified_number

        #         Individual.fail
        # in initialization or after every mutation of all population

        # Prediction

        batch_individual = np.array([ind.indices for ind in self.population_to_mutate])
        batch_label = [ind.expected_label for ind in self.population_to_mutate]

        if self.xai_method == "Lime":
            explanation = lime_batch_explainer(predictor.predict_texts_xai, batch_individual)
        elif self.xai_method == "Random":
            # assgin random attention map for random method
            explanation = np.random.rand(len(batch_individual), MAX_SEQUENCE_LENGTH) #[1] * len(batch_individual)
        else:
            explanation = xai_embedding(predictor.model,
                                        batch_individual,
                                        xai_method=self.xai_method,
                                        target_class=batch_label)

        predictions, confidences = (predictor.predict(batch_individual))
        # label result and detect misclassified
        for ind, prediction, confidence, attmap \
                in zip(self.population_to_mutate, predictions, confidences, explanation):
            ind.confidence = confidence
            ind.predicted_label = prediction
            ind.misclassified = ind.expected_label != ind.predicted_label

            ind.attention = attmap
            # TODO: enable it if needed
            if MUTATION_RECORD:
                ind.append_mutation_log(gen_number, folder)

        for ind in self.population_to_mutate:
            if ind.misclassified or ind.fail:
                self.population_to_mutate.remove(ind)
                ind.mutate_attempts = gen_number
                self.finished_population.append(ind)
                if ind.misclassified:
                    self.misclassified_number += 1
                    ind.export(folder.archive_folder )
                # if MUTATION_RECORD:
                #     ind.save_mutation_log(folder)


    def mutate_population(self):
        batch_individual = [ind for ind in self.population_to_mutate]

        for ind in batch_individual:
            if self.xai_method == "Lime":
                status, tokens = mutate_lime(ind.pure_indices, ind.attention, ind.expected_label)
            else:
                unpad_indices, pad_length = utils.remove_padding(ind.indices)
                unpad_explanation = ind.attention[pad_length:]

                status, tokens = mutate(unpad_indices, unpad_explanation, self.xai_method)
            if not status:
                ind.fail = True
                continue
            ind.pure_indices = tokens
            ind.reset()
            # TODO mark

    def create_report(self, path, gen_number, seed):
        dst = join(path, REPORT_NAME)
        with open(dst, mode='w') as report_file:
            report_writer = csv.writer(report_file,
                                       delimiter=',',
                                       quotechar='"',
                                       quoting=csv.QUOTE_MINIMAL)
            report_writer.writerow(['population size',
                                    'total iteration number',
                                    'misbehaviour number'])
            report_writer.writerow([self.size,
                                    gen_number,
                                    self.misclassified_number
                                    ])

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
                                        ind.misclassified,
                                        ind.confidence,
                                        ind.mutate_attempts,
                                        ind.fail])

        # summary table
        table_name = './runs/summary.csv'

        data = [self.size,
                gen_number,
                self.xai_method,
                self.misclassified_number,
                "{:.2f}%".format(self.misclassified_number/self.size*100),
                seed]

        # create csv if it's not existed
        if not exists(table_name):
            with open(table_name, 'w', newline='') as file:
                writer = csv.writer(file)
                headers = ['population size',
                           'total iteration number',
                           'attention method',
                           'misbehaviour number',
                           'misbehaviour Percentage',
                           'seed']
                writer.writerow(headers)
                writer.writerow(data)
        else:
            # only write data when csv is already existed
            with open(table_name, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)


if __name__ == "__main__":
    pop_size = 3

    pop = Population(pop_size=pop_size)
    print(f" Population size {pop.size}")
    # pop.evaluate_population(0)

    for idx in range(2):
        pop.evaluate_population(idx)
        print([ind.confidence for ind in pop.population_to_mutate])
        pop.mutate_population()


    print(f" Misclass number {pop.misclassified_number}")
