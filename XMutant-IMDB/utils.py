from babel.plural import in_range_list
from nltk.corpus import wordnet
import random
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import string
import os
import csv
from config import DEFAULT_WORD_ID, INDEX_FROM, NUM_DISTINCT_WORDS, MAX_SEQUENCE_LENGTH
import difflib

WORD_TO_ID = imdb.get_word_index(path="imdb_word_index.json")
WORD_TO_ID = {k: (v + INDEX_FROM) for k, v in WORD_TO_ID.items()}
WORD_TO_ID.update(DEFAULT_WORD_ID)
# WORD_TO_ID["<pad>"] = 0
# WORD_TO_ID["<start>"] = 1
# WORD_TO_ID["<unk>"] = 2
# WORD_TO_ID["<unused>"] = 3
ID_TO_WORD = {value: key for key, value in WORD_TO_ID.items()}


def replace_multiple_spaces(sentence):
    return re.sub(r'\s+', ' ', sentence)


def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))


def pad_inputs(indices):
    # 0.0 because it corresponds with <PAD>
    return pad_sequences(indices, maxlen=MAX_SEQUENCE_LENGTH, value = DEFAULT_WORD_ID["<pad>"])

def remove_padding(padded_sequence):
    """
    For one vector, remove all padding values
    """
    # Trim padding_value from the start, pad values are 0
    trim_sequence = np.trim_zeros(padded_sequence, 'f')
    trim_length = len(padded_sequence) - len(trim_sequence)
    return trim_sequence, trim_length

def unpad_inputs(padded_sequences):
    """
    For one vector, remove all padding values
    """
    unpadded_sequences = []
    for seq in padded_sequences:
        # Trim padding_value from the end
        trim_sequence, _ = remove_padding(seq)
        unpadded_sequences.append(trim_sequence)
    return unpadded_sequences

def indices2words(id_list):
    # if len(id_list) < MAX_SEQUENCE_LENGTH:
    #     id_list = pad_inputs([id_list])[0]
    return ' '.join(ID_TO_WORD[id] for id in id_list)


def words2indices(word_list, num_distinct_words = NUM_DISTINCT_WORDS):
    word_list = word_list.lower()
    word_list = remove_punctuation(word_list)
    word_list = replace_multiple_spaces(word_list)

    indices = []
    for word in word_list.split(' '):
        try:
            id = WORD_TO_ID[word]
            if id >= num_distinct_words:
                indices.append(DEFAULT_WORD_ID["<unk>"])
            else:
                indices.append(id)
        except KeyError:
            # print("Unknown word:" + word)
            indices.append(DEFAULT_WORD_ID["<unk>"])
    return indices


def csv_logger(filepath: str, log_info: dict):
    if not filepath.endswith(".csv"):
        filepath += ".csv"

    # write CSV to file
    if not os.path.exists(filepath):
        # create csv file header
        with open(filepath, 'w', encoding='UTF8') as f:
            writer = csv.writer(f,
                                delimiter=',',
                                quotechar='"',
                                quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')
            # write the header
            writer.writerow(list(log_info.keys()))

    with open(filepath, 'a', encoding='UTF8') as f:
        writer = csv.writer(f,
                            delimiter=',',
                            quotechar='"',
                            quoting=csv.QUOTE_MINIMAL,
                            lineterminator='\n')
        writer.writerow(list(log_info.values()))

def set_all_seeds(digit):
    import tensorflow as tf
    random.seed(digit)
    np.random.seed(digit)
    tf.keras.utils.set_random_seed(digit)

def find_word_location(sentence, word):
    # Split the sentence into words
    words = sentence.split()

    # Use difflib to find the closest match in the sentence
    closest_matches = difflib.get_close_matches(word, words, n=1, cutoff=0.8)

    # If a close match is found, return its position
    if closest_matches:
        return words.index(closest_matches[0])
    else:
        return -1
