import csv
import utils
import predictor
import time
import json
import os

class Individual:
    def __init__(self, id, token_ids, label):
        self.id = id
        self.pure_indices = token_ids
        self.expected_label = label
        # reset
        self.predicted_label = None
        self.confidence = None
        self.indices = utils.pad_inputs([self.pure_indices])[0]
        self.text = utils.indices2words(self.indices)
        self.attention = None

        self.misclassified: bool = False
        self.mutate_attempts = None
        self.mutation_point = (None, None)
        self.fail: bool = False
        self.timestamp = time.time()

    def reset(self):
        self.predicted_label = None
        self.confidence = None
        self.indices = utils.pad_inputs([self.pure_indices])[0]
        self.text = utils.indices2words(self.indices)
        self.attention = None

    def export(self, json_folder):

        if self.predicted_label is None:
            # TODO: check
            self.predicted_label = predictor.predict_single_text(self.text)[0][0]

        log_info = {'id': str(self.id),
                'expected_label': str(self.expected_label),
                'predicted_label': str(self.predicted_label),
                'misbehaviour': str(self.misclassified),
                'confidence': str(self.confidence),
                'timestamp': str(self.timestamp),
                'elapsed': str(time.time() - self.timestamp),
                'token': ' '.join([str(x) for x in self.pure_indices]),
                'text': utils.indices2words(self.pure_indices)
                }

        file_path = json_folder + '/' + str(self.id) + '.json'

        with open(file_path, 'w') as f:
            (json.dump(log_info, f, sort_keys=True, indent=4))

    def append_mutation_log(self, gen_number, folder):
        dst = os.path.join(folder.individual_folder, 'ID'+str(self.id).zfill(4)+".csv")

        data_to_append = {"id": self.id,
                          "generation": gen_number,
                          "expected_label": self.expected_label,
                          "predicted_label": self.predicted_label,
                          "misbehaviour": self.misclassified,
                          "confidence": self.confidence,
                          "text": utils.indices2words(self.pure_indices),
                          }

        utils.csv_logger(dst, data_to_append)