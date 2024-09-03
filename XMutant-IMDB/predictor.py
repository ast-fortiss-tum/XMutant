# For Python 3.6 we use the base keras
import tensorflow as tf
from numpy.ma.core import indices
from tensorflow import keras
import pickle
# from properties import MODEL, EXPECTED_LABEL
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.datasets import imdb
import re
import string

# from properties import INPUT_MAXLEN
from config import (MAX_SEQUENCE_LENGTH, NUM_DISTINCT_WORDS,MODEL,
                    EMBEDDING_OUTPUT_DIMS,DEFAULT_WORD_ID, INDEX_FROM)
from utils import (#WORD_TO_ID, ID_TO_WORD, indices2words,
                   words2indices, pad_inputs
                  )

class Predictor:

    def __init__(self, model_dir=MODEL,
                     max_sequence_length = MAX_SEQUENCE_LENGTH,
                     num_distinct_words = NUM_DISTINCT_WORDS):
        self.model = tf.keras.models.load_model(model_dir)
        print(f"Loaded model from disk {model_dir}")
        self.max_sequence_length = max_sequence_length
        self.num_distinct_words = num_distinct_words

    def predict_single_text(self, text):
        # Tokenize
        assert type(text) == str, "input must be a string"
        indices = words2indices(text)
        predictions, confidences = self.predict([indices])
        return predictions, confidences

    def predict_text_list(self, text_list):
        # Tokenize
        index_list = [words2indices(text) for text in text_list]
        predictions, confidences = self.predict(index_list)
        return predictions, confidences

    def predict(self, index_list):
        index_list = pad_inputs(index_list)
        # Predictions vector
        predictions_output = self.model.predict(index_list)
        predictions_output = predictions_output.reshape(predictions_output.shape[0])

        predictions = (predictions_output >= 0.5).astype(int)
        confidences = np.abs(1-predictions_output-predictions)

        return predictions, confidences

    def predict_texts_xai(self, text_list):
        # Tokenize
        index_list = [words2indices(text) for text in text_list]
        # pad inputs
        index_list = pad_inputs(index_list)
        # Predictions vector
        # if index_list.max() >= NUM_DISTINCT_WORDS:
        #     print(index_list.max())
        predictions = self.model.predict(index_list)
        if predictions.shape[1] == 1:
            predictions = np.hstack([1 - predictions, predictions])
            # print(predictions.shape)  # Ensure this prints (num_samples, 2)
        return predictions

    def predict_single_text_xai(self, text):
        # Tokenize
        index_list = [words2indices(text)]
        # pad inputs
        index_list = pad_inputs(index_list)
        # Predictions vector
        # if index_list.max() >= 10000:
        #     print(index_list.max())
        predictions = self.model.predict(index_list)
        if predictions.shape[1] == 1:
            predictions = np.hstack([1 - predictions, predictions])
            #print(predictions.shape)  # Ensure this prints (num_samples, 2)
        return predictions

    def predict_tabular_xai(self, index_list):
        # Tokenize
        # pad inputs

        index_list = pad_inputs(index_list)
        index_list = np.clip(index_list, 0, NUM_DISTINCT_WORDS - 1)
        # Predictions vector
        predictions = self.model.predict(index_list)
        if predictions.shape[1] == 1:
            predictions = np.hstack([1 - predictions, predictions])
            print(predictions.shape)  # Ensure this prints (num_samples, 2)
        return predictions