from affinity_aglorithms.utils.files import pickle_load, pickle_dump
from affinity_aglorithms.metrics import semantic_shift
import config
import os
import sys


def sort_aff_words(affinity_terms, n=1000, reverse=False):
    """Sorts affinity terms in order specified.
    
    Args:
        affinity_terms (dic): dictionary of terms with affinity scores
        n (int): number of terms to extract
        reverse (bool): Order in which to extract. False is High, True is Neutral.
        
    Returns:
        
    """
    aff_words = sorted(affinity_terms, key=lambda x: affinity_terms[x], reverse=reverse)
    if reverse:
        aff_n = aff_words[:n]
    else:
        aff_words = [aff for aff in aff_words if affinity_terms[aff] != 0]
        aff_n = aff_words[:n]
    return aff_n


def load_affinity_terms(sub_space):
    """Loads affinity terms from a provided path.
    
    Args:
        sub_space (str): Path for subreddit space
    
    Returns:
        Loads affinity terms from the specified directory and returns it.
    """
    if affinity_terms_path == "default":
        aff_path = subreddit + "_affinity_terms.pkl"
        affinity_terms_path = os.path.join(sub_space, aff_path)
    affinity_terms = pickle_load(affinity_terms_path)
    return affinity_terms


def run_embedding_affinity_scores(subreddit,
                                  n=1000, 
                                  reverse=False):
    """Extracts affinity scores for specified subreddits.  
    
    Args:
        subreddit (str): Subreddit name
        n (int): number of affinity words to extract
        reverse (bool): Order of affinity words to extract. False is high, True is neutral
        
    State:
        dumps specified file extraction to the passed directory.
    """
    sub_space = os.path.join(config.SUBDIR_ANALYSIS_LOAD_PATH, subreddit)
    affinity_terms = load_affinity_terms(sub_space)
    
    aff_n = sort_aff_words(affinity_terms)
        
    _, word_to_aff_score = semantic_shift.get_n_affinity_words_semantic_shift(subreddit, aff_n)
    if reverse:
        save_str = '_aff2_' + str(n) + '.pkl'
    else:
        save_str = '_neutral2_' + str(n) + '.pkl'
    pickle_dump(subreddit, save_str, sub_space, word_to_aff_score)
    print(subreddit)


if __name__ == '__main__':
    """Main method is invoked for when running with GNU parallel through cmdline.
    
    Args:
        sys.argv[1]: subreddit name
    """
    subreddit = sys.argv[1]
    run_embedding_affinity_scores(subreddit)
