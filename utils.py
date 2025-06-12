import numpy as np
import shutil
import random
from copy import deepcopy
from copy import copy
import string

def az_list():
    """
    Returns a default a-zA-Z characters list
    """
    cx = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    return cx

def generate_random_permutation_map(chars):
    """
    Generate a random permutation map for given character list. Only allowed permutations
    are training_dictical ones. Helpful for debugging
    
    Arguments:
    chars: list of characters
    
    Returns:
    p_map: a randomly generated permutation map for each character
    """
    cx = az_list()
    cx2 = az_list()
    random.shuffle(cx2)
    p_map = generate_identity_p_map(chars)
    for i in range(len(cx)):
        p_map[cx[i]] = cx2[i]
        
    return p_map
    
def generate_identity_p_map(chars):
    """
    Generates an identity permutation map for given list of characters
    
    Arguments:
    chars: list of characters
    
    Returns:
    p_map: an identity permutation map
    
    """
    p_map = {}
    for c in chars:
        p_map[c] = c
    
    return p_map
    
def scramble_text(text, p_map):
    """
    Scrambles a text given a permutation map
    
    Arguments:
    text: text to scramble, list of characters
    
    p_map: permutation map to scramble text based upon
    
    Returns:
    text_2: the scrambled text
    """
    text_2 = []
    for c in text:
        text_2.append(p_map[c])
        
    return text_2
    
def shuffle_text(text, i1, i2):
    """
    Shuffles a text given the index from where to shuffle and
    the upto what we should shuffle
    
    Arguments:
    i1: index from where to start shuffling from
    
    i2: index upto what we should shuffle, excluded.
    """
    
    y = text[i1:i2]
    random.shuffle(y)
    t = copy(text)
    t[i1:i2] = y
    return t
    
def move_one_step(p_map):
    """
    Swaps two characters in the given p_map
    
    Arguments:
    p_map: A p_map
    
    Return:
    p_map_2: new p_map, after swapping the characters
    """
    
    keys = az_list()
    sample = random.sample(keys, 2)
    
    p_map_2 = deepcopy(p_map)
    p_map_2[sample[1]] = p_map[sample[0]]
    p_map_2[sample[0]] = p_map[sample[1]]
    
    return p_map_2

def move_one_step_but_better(p_map, text, frequency_statistics, char_to_ix):

    #accessing the relative frequency of all 26 letters in the training text
    training_dict = [c for c in char_to_ix.keys() if c in string.ascii_letters]

    #Only grabbing charcters from the decrypted text
    filtered = [c for c in text if c in training_dict]
    total = len(filtered)

    # decrypting the filtered text
    decrypted = [p_map[c] for c in filtered]

    # calculating observed frequencies over training text
    current_counts = np.zeros(len(training_dict))
    for c in decrypted:
        if c in training_dict:
            current_counts[training_dict.index(c)] += 1
    current_rel = current_counts / total

    # building expected frequencies across the training_dict
    total_expected = np.sum(frequency_statistics)
    expected_rel = np.array([frequency_statistics[char_to_ix[c]] for c in training_dict]) / total_expected

    # identifying descrepancies between current and expected frequencies of letters
    diffs = np.abs(current_rel - expected_rel)

    # creating a distribution of weights for each proposal - spanning all pairs of letters
    pairs, weights = [], []
    n = len(training_dict)
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((training_dict[i], training_dict[j]))
            weights.append(diffs[i] + diffs[j])
    weights = np.array(weights)

    # normalize the weights to create a probability distribution
    probs = weights / weights.sum()
    # sample a pair of letters based on the calculated distribution
    idx = random.choices(range(len(pairs)), weights = probs, k = 1)[0]
    c1, c2 = pairs[idx]

    # applying the swap in the previous p_map
    new_p_map = deepcopy(p_map)
    new_p_map[c1], new_p_map[c2] = p_map[c2], p_map[c1]
    return new_p_map





def pretty_string(text, full=False):
    """
    Pretty formatted string
    """
    if not full:
        return ''.join(text[1:200]) #+ shutil.get_terminal_size().columns*'-'#'...'
    else:
        return ''.join(text) #+ shutil.get_terminal_size().columns*'-'#'...'

def compute_statistics(filename):
    """
    Returns the statistics for a text file.
    
    Arguments:
    filename: name of the file
    
    Returns:
    char_to_ix: mapping from character to index
    
    ix_to_char: mapping from index to character
    
    transition_probabilities[i,j]: gives the probability of j following i, smoothed by laplace smoothing
    
    frequency_statistics[i]: gives number of times character i appears in the document
    """
    data = open(filename, 'r').read() # should be simple plain text file
    chars = list(set(data))
    N = len(chars)
    char_to_ix = {c : i for i, c in enumerate(chars)}
    ix_to_char = {i : c for i, c in enumerate(chars)}
    
    transition_matrix = np.ones((N, N))
    frequency_statistics = np.zeros(N)
    i = 0
    while i < len(data)-1:
        c1 = char_to_ix[data[i]]
        c2 = char_to_ix[data[i+1]]
        transition_matrix[c1, c2] += 1
        frequency_statistics[c1] += 1
        i += 1
        
    frequency_statistics[c2] += 1
    transition_matrix /= np.sum(transition_matrix, axis=1, keepdims=True)
    
    return char_to_ix, ix_to_char, transition_matrix, frequency_statistics