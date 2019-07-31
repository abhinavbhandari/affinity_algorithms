# import pandas as pd

from subreddinfo import pickle_load, pickle_dump
import multiprocessing as mp
import sys
import os
import config

"""A global object that holds the word-to-subreddits dictionary with each words frequency. """
global word_to_subs_dic_global 
word_to_subs_dic_global = pickle_load(os.path.join(config.SUBREDDIT_METRIC, "word_frequencies_to_subreddit.pkl"))


def get_n_most_affinity_terms(aff_dic, n=100, reverse=False):
    """Return top n affinity terms."""
    terms = sorted(aff_dic, key=lambda x: aff_dic[x], reverse=reverse)
    if reverse==False:
        terms = [term for term in terms if aff_dic[term] != 0]
    return terms[:n]


def get_n_most_affinity_terms_mult(aff_dics, n=100, reverse=False):
    """Helper function that gets top n affinity terms (for multiple subreddits).
    
    Args:
        aff_dics: list of dictionaries with words that map to affinity values. 
            Each dictionary represents a subreddit
        n: Number of affinity terms to extract from highest to lowest.
        
    Returns:
        n affinity terms in descending order. 
    """
    terms = []
    for aff_dic in aff_dics:
        terms.append(get_n_most_affinity_terms(aff_dic, n, reverse))
    return terms


def get_word_to_subs_dic(word):
    """ Return list of subreddits and their frequencies for a word"""
    return word_to_subs_dic_global[word]


def get_wordfreq_dic(subname, wordfreq_ext='_filtered_lemma.pkl'):
    """Return word-frequency-dictionary for a subreddit."""
    lemma_dic_path = subname + wordfreq_ext
    sub_lemma_dic_path = os.path.join(
        os.path.join(config.SUBDIR_ANALYSIS_LOAD_PATH, subname), 
        lemma_dic_path)
    return pickle_load(sub_lemma_dic_path)


def calculate_total_freq(subnames, mult=False):
    """ Calculate the total number of occurrences of all terms in a subreddit.
    
    Calculates the total word occurrence in subreddits passed to the function.
    Has functionality to calculate for multiple subreddits and single subreddit.
    
    Args:
        subnames: a list of subreddits
        mult: boolean for multiple file processing or not.
    
    Returns:
        An int that is a sum, or a list of sum of word occurrences. 
    
    """
    if mult:
        freq = []
        for sub in subnames:
            dic = get_wordfreq_dic(sub)
            freq.append(sum(v for _, v in dic.items()))
        return freq
    else:
        return sum(v for (_, v) in dics.items())
    

def affinity_analysis(subnames, target_sub_index, total_freq_path="total_freq.pkl", subs_to_num_path="subs_to_num.pkl"):
    """Computes the affinity values of terms in a subreddit.
    
    The affinity value is computed in comparison to the total frequency of terms from other subreddits.
    
    Fomrula for the affinity function is the following.
    (1 - 1/(f_s(w) - t_f(s) * 0.00001)) * ( p_s(w)/SUM for s' in S(p_s'(w)) )
    
    Args:
        wordfreq_dic_list: a list of wordfreq dictionaries, for each subreddit.
        total_freq: a dictionary that contains the total occurrences for each subreddit.
        subs_to_num: a dictionary that maps a subreddit to number index
        target_sub_index: this is the index of the target subreddit for which 
                affinity terms are being calculated. 
    
    Returns:
        A dictionary that maps affinity value to each word in the target subreddit.
    """
    if total_freq_path:
        total_freq = pickle_load(os.path.join(config.SUBREDDIT_METRIC, total_freq_path))
    else:
        total_freq = calculate_total_freq(subnames, mult=True)
    subs_to_num = pickle_load(os.path.join(config.SUBREDDIT_METRIC, subs_to_num_path))
    target_sub_word_freq_dictionary = get_wordfreq_dic(subnames[target_sub_index])
    affinity = []
    affinity_value_dictionary = {}
    target_dictionary_index = subs_to_num[subnames[target_sub_index]]
    for word in target_sub_word_freq_dictionary:
        a = target_sub_word_freq_dictionary[word]/total_freq[target_dictionary_index]
        word_to_subs_list = get_word_to_subs_dic(word)
        for j, sub in enumerate(subnames):
            sub_index = subs_to_num[sub]
            if j != target_sub_index and sub_index in word_to_subs_list:
                a += word_to_subs_list[sub_index]/total_freq[sub_index]
        fp = int(total_freq[target_dictionary_index]*0.00001)
        coeff = target_sub_word_freq_dictionary[word] - fp
        if coeff <= 1:
            coeff = 1
        num = (1 - 1/coeff)*((target_sub_word_freq_dictionary[word]/total_freq[target_dictionary_index])/a)
        affinity_value_dictionary[word] = num
    return affinity_value_dictionary


def run_slang_analyser(subnames, word_to_subs_global_path="word_frequencies_to_subreddit.pkl", total_freq_path="total_freq.pkl", subs_to_num_path="subs_to_num.pkl", save_name='temporary', save_file=False):
    """Runs slang analysis, which includes calculating affinity values for each subreddit passed.
    
    This function is the main function that encapsulates the logic for all the steps required to run
    affinity analysis. The affinity analysis should extract key terms. 
    
    Args:
        subnames: a list of subreddit names, of which directories should be accessed and their lemma dictionaries
                should be used to extract high affinity terms, relative to other subreddits.
        target_sub: a subreddit that should be loaded, separetly,and its file should be loaded from disc.
        
    Returns:
        a list which contains a dictionary of affinity values for each word. 
    """
    
    num_of_processes = os.cpu_count() - 1
    pool = mp.Pool(processes=num_of_processes)
    
    affinity_dic_list = []
    for i_position in range(len(subnames)):
        affinity_dic_list.append(pool.apply(affinity_analysis, args=(subnames, i_position, total_freq_path, subs_to_num_path)))
    
    if save_file:
        affinity_values_save_path = os.path.join(config.SUBDIR_ANALYSIS_LOAD_PATH, 'affinity_values')
        if not os.path.exists(affinity_values_save_path):
            os.makedirs(affinity_values_save_path)
        pickle_dump(save_name, '_affinity_terms.pkl', affinity_values_save_path, affinity_dic_list)
    else:
        return affinity_dic_list
    
    
def load_affinity_terms(sub):
    """ """ 
    aff_terms_ext = '_affinity_terms.pkl'
    sub_space = os.path.join(config.SUBDIR_ANALYSIS_LOAD_PATH, sub)
    aff_terms_file_path = sub + aff_terms_ext
    full_path = os.path.join(sub_space, aff_terms_file_path)
    return pickle_load(full_path)
    
    
def load_affinity_terms_mult(subnames):
    """ """
    aff_terms_list = []
    for sub in subnames:
        aff_terms_list.append(load_affinity_terms(sub))
    return aff_terms_list
        

if __name__ == '__main__':
    """
    Hello hello
    """
    subname_file = sys.argv[1]
    
    with open(subname_file) as f:
        subnames = f.read()
    print(type(subnames))
    subnames = subnames.split('\n')[:-1]
    print(len(subnames))
    run_slang_analyser(subnames)