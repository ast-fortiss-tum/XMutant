import random
import logging as log

from xai_imdb import top_k_attributions
import sys
# log.basicConfig(stream=sys.stdout, level=log.DEBUG)
import numpy as np
from utils import indices2words, words2indices, ID_TO_WORD, WORD_TO_ID, find_word_location
from nltk.corpus import wordnet
import nltk


def mutate(tokens, attributions, xai_method):
    """
    Mutate a list of word indices based on their weights

    Input:
    token: list of unpadded word indices
    weights: list of word explanations,

    """

    assert len(tokens) == len(attributions), "tokens and weights must have the same length"
    assert tokens[0] != 0, "tokens must be unpadded"

    # obtain the top k words based on attributions
    word_indices, sorted_attributions, locations = top_k_attributions(tokens, attributions)

    # mutation_method = random.choice([1,2])
    mutation_methods = [mut1, mut2]
    random.shuffle(mutation_methods)
    for med in mutation_methods:
        status, new_token, location = med(word_indices, sorted_attributions, locations, tokens, xai_method)
        if status:
            break

    if status:
        # insert new tokens at the specified location
        # tokens[location] = new_token[0]
        # if len(new_token) > 1:
        #     tokens.insert(location + 1, new_token[1:])
        log.info(f"Before Mutation: {indices2words(tokens)}")
        if len(tokens) == location + 1:
            tokens = np.concatenate((tokens[:location], new_token))
        elif location == 0:
            tokens = np.concatenate((new_token, tokens[1:]))
        else:
            tokens =np.concatenate((tokens[:location], new_token, tokens[location+1:]))
        log.info(f"After Mutation: {indices2words(tokens)}")
        return status, tokens

    return status, tokens

def mutate_lime(tokens, explanations, label):
    """
    Mutate a list of word indices based on their weights

    Input:
    token: list of unpadded word indices
    explanations: lengthx2 list of (index, attribution)

    """
    assert tokens[0] != 0, "tokens must be unpadded"
    if not isinstance(tokens, np.ndarray):
        tokens = np.array(tokens)

    # # remove index 0 (<pad>) in explanations
    # explanations = np.delete(explanations, np.where(explanations[:,0] == 0), axis=1)
    # explanations = np.delete(explanations, np.where(explanations[:,0] == 6887), axis=1) # pad
    # word_indices = explana[tions[:,0].astype(int)
    # sorted_attributions = explanations[:,1]

    word_indices = []
    sorted_attributions = []
    locations = []
    # Iterate over each selected token
    for i, content  in enumerate(explanations):
        # Find the indices where the token occurs
        index = int(content[0])
        if index == 0 or index == 6887 or index == 380: # pad start
            continue
        location = np.where(tokens == index)[0]

        # If the token exists in the array, get the last occurrence
        if location.size > 0:
            locations.append(location[-1])
        else:
            location = find_word_location(indices2words(tokens), ID_TO_WORD[index]) # log.info(f"Token {index} not found in tokens: {tokens}")

            if location == -1 or location == 0:
                # print(f"Token {index}-{ID_TO_WORD[index]} not found in tokens: {tokens}-{indices2words(tokens)}")
                continue
            else:
                locations.append(location)
        word_indices.append(index)
        if label == 1:
            sorted_attributions.append(content[1])
        elif label == 0:
            sorted_attributions.append(-content[1])
        else:
            raise Exception(f"Invalid label {label}")

    word_indices = np.array(word_indices).astype(int)
    sorted_attributions = np.array(sorted_attributions)
    locations = np.array(locations).astype(int)

    mutation_methods = [mut1, mut2]
    random.shuffle(mutation_methods)
    for med in mutation_methods:
        status, new_token, location = med(word_indices, sorted_attributions, locations, tokens, xai_method="Lime")
        if status:
            break

    if status:
        # insert new tokens at the specified location
        # tokens[location] = new_token[0]
        # if len(new_token) > 1:
        #     tokens.insert(location + 1, new_token[1:])
        log.info(f"Before Mutation: {indices2words(tokens)}")
        if len(tokens) == location + 1:
            tokens = np.concatenate((tokens[:location], new_token))
        elif location == 0:
            tokens = np.concatenate((new_token, tokens[1:]))
        else:
            tokens =np.concatenate((tokens[:location], new_token, tokens[location+1:]))

        log.info(f"After Mutation: {indices2words(tokens)}")
        return status, tokens

    return status, tokens


# replace a word with its synonym
def apply_mutoperator1(tokens, attributions, locations):
    """
    Replace a word with its synonym
    Input:
    - tokens: list of word indices
    - attributions: list of word explanations
    - locations: list of word locations
    Output:
    - status: success or not
    - new_tokens: new word indices
    - new_location: location of the new word
    """
    if len(tokens) == 0:
        return False, None, None
    assert len(tokens) == len(attributions), "tokens and weights must have the same length"
    assert len(tokens) == len(locations), "tokens and locations must have the same length"
    if not isinstance(tokens, list):
        tokens = list(tokens)
    if not isinstance(locations, list):
        locations = list(locations)

    weights = np.abs(attributions)
    weights = list(weights)
    while len(tokens) > 0:
        # rand_index = random.randint(0, len(vector) - 1)
        selected_token = random.choices(population=tokens, weights=weights, k=1)[0]
        selected_index = tokens.index(selected_token) # tokens.index(selected_token)

        selected_word = ID_TO_WORD[selected_token]

        syn = get_synonym(selected_word) # can be None

        if syn is not None:
            new_token = words2indices(syn) # a list

            log.info(f"replace a word with its synonym: {selected_word} => {syn}")
            return True, new_token, locations[selected_index]

        # Did not find a synonym, remove the selected word from the list

        tokens.pop(selected_index)
        weights.pop(selected_index)
        locations.pop(selected_index)

    return False, None, None

def apply_mutoperator2(tokens, attributions, locations):
    """
    Duplicate a word with its synonym
    """
    if len(tokens) == 0:
        return False, None, None
    assert len(tokens) == len(attributions), "tokens and weights must have the same length"
    assert len(tokens) == len(locations), "tokens and locations must have the same length"

    if not isinstance(tokens, np.ndarray):
        tokens = np.array(tokens)
    if not isinstance(locations, np.ndarray):
        locations = np.array(locations)

    weights = np.abs(attributions)

    words = [ID_TO_WORD[id] for id in tokens]
    adjs, adjs_index = find_adj_adv(words)

    # check if we have any adj in the mutation candidates
    while len(adjs_index) > 0:
        selected_weights = weights[adjs_index]

        selected_index = random.choices(population=adjs_index, weights=selected_weights, k=1)[0]

        selected_word = words[selected_index]
        syn = get_synonym(selected_word)

        if syn is not None:
            new_token = words2indices(selected_word + " and " + syn) # a list

            log.info(f"insert a synonym: {selected_word} => {selected_word + ' and ' + syn}")
            return True, new_token, locations[selected_index]
        # Did not find a synonym, remove the selected word from the list

        adjs_index.remove(selected_index)
        # selected_weights = np.delete(selected_weights, np.where(selected_weights == weights[selected_index])[0][0])

    return False, None, None


def mut1(word_indices, sorted_attributions, locations, tokens=None, xai_method=None):
    status, new_token, location = apply_mutoperator1(word_indices, sorted_attributions, locations)
    return status, new_token, location

def mut2(word_indices, sorted_attributions, locations, tokens, xai_method):
    # remove if the location the next to 'and'
    to_remove = []
    # Iterate through the select list in reverse to safely remove elements
    for i, loc in enumerate(locations):  # use select[:] to avoid modifying the list while iterating
        if (loc > 0 and tokens[loc - 1] == 5) or (loc < len(tokens) - 1 and tokens[loc + 1] == 5):
            to_remove.append(i)
    if len(to_remove) > 0:
        word_indices = np.delete(word_indices, to_remove)
        sorted_attributions = np.delete(sorted_attributions, to_remove)
        locations = np.delete(locations, to_remove)

    if xai_method == "SmoothGrad" or xai_method == "Random":
        status, new_token, location = apply_mutoperator2(word_indices, sorted_attributions, locations)
    elif xai_method == "IntegratedGradients" or xai_method == "Lime":
        neg_token = word_indices[sorted_attributions < 0]
        neg_attributions = sorted_attributions[sorted_attributions < 0]
        neg_locations = locations[sorted_attributions < 0]
        # prioritize negative attributions for mutation 2
        status, new_token, location = apply_mutoperator2(neg_token, neg_attributions, neg_locations)
    else:
        raise Exception(f"Invalid xai method {xai_method}")
    return status, new_token, location

def get_synonym(word):
    word = word.lower()
    synonyms = []
    synsets = wordnet.synsets(word)
    if (len(synsets) == 0):
        return None
    for synset in synsets:
        lemma_names = synset.lemma_names()
        for lemma_name in lemma_names:
            lemma_name = lemma_name.lower().replace('_', ' ')
            if (lemma_name != word and lemma_name not in synonyms):
                synonyms.append(lemma_name)
    if len(synonyms) == 0:
        return None
    else:
        while len(synonyms) >0:
            sword = random.choice(synonyms)
            if word in sword:
                synonyms.remove(sword)
                continue
            elif WORD_TO_ID["<unk>"] in words2indices(sword):
                synonyms.remove(sword)
                continue
            else:
                return sword
        return None



def find_adj_adv(words_list):
    word_tags = nltk.pos_tag(words_list)
    # print(word_tags)
    adjs_advs = []
    ad_id = []
    for i in range(0,len(word_tags)):
        if word_tags[i][1] in ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']:
            adjs_advs.append(word_tags[i][0])
            ad_id.append(i)
    return adjs_advs, ad_id




if __name__ == "__main__":
    # indices = np.array([291, 7, 4, 20, 118, 17, 31, 4, 7,
    #        2794, 14, 39, 112, 12744, 967, 972, 13, 35,
    #        327, 10509])
    # weights = np.array([4.2238919e-04, 9.3142633e-05, 7.6749886e-05, 6.4159132e-05,
    #                    5.7167133e-05, 4.3338347e-05, 4.2995649e-05, 4.1660820e-05,
    #                    3.9781873e-05, 3.7703692e-05, 3.4348061e-05, 3.2243166e-05,
    #                    3.2177235e-05, 3.0968287e-05, 3.0458821e-05, 2.7641205e-05,
    #                    2.5889685e-05, 2.3884184e-05, 2.3466435e-05, 2.2520195e-05])
    # locations = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

    indices = np.array([19, 1580, 2354, 8, 2579, 3916, 2835, 157, 9494, 17, 5704, 713, 7516, 8844, 1639, 22, 4650, 37, 4611, 12459,
                        38])
    weights = np.array([0.4539333, 0.25433126, 0.20099907, 0.19521411, 0.19221331, 0.18454358, 0.18154374,
                        0.17476512, 0.1699163, 0.15491284, 0.1545332, 0.15090293, 0.14655387,
                        0.14452878, 0.14290535, 0.14032906, 0.13923967, 0.13889316, 0.13619153, 0.1338239,
                        0.5])

    # [207 180 203 205  41 138 204 206 134 190 170 197 188  63 151 179 122  44 189 147]
    locations = np.array([207, 180, 203, 205, 41, 138, 204, 206, 134, 190, 170, 197, 188, 63, 151, 179, 122, 44, 189, 147,
                          45])
    for i in range(10):
        # apply_mutoperator1(indices, weights, locations)
        apply_mutoperator2(indices, weights, locations)