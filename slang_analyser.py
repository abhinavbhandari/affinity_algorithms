# import pandas as pd

from subreddinfo import pickle_load, pickle_dump
import multiprocessing as mp
import sys
import os
import config

def get_n_most_affinity_terms(aff_dic, n=100):
    terms = sorted(aff_dic, key=lambda x: aff_dic[x], reverse=True)
#     print(type(terms))
    return terms[:n]


def get_n_most_affinity_terms_mult(aff_dics, n=100):
    """
    Helper function that gets 
    """
    terms = []
    for aff_dic in aff_dics:
        terms.append(get_n_most_affinity_terms(aff_dic, n))
    return terms


def calculate_total_freq(dics, mult=False):
    """
    Purpose:
        dics: A list of dics of lemma. 
    
    """
    if mult:
        freq = []
        for dic in dics:
            freq.append(sum(v for _, v in dic.items()))
        return freq
    else:
        return sum(v for (_, v) in dics.items())


def affinity_analysis(wordfreq_dic_list, total_freq, ignore_pos):
    """
    Purpose:
        
    
    """
    main_vocab_dic = wordfreq_dic_list[ignore_pos]
    affinity = []
    holder = {}
    for word in main_vocab_dic:
        a = main_vocab_dic[word]/total_freq[ignore_pos]
        for j in range(len(wordfreq_dic_list)):
            if j is not ignore_pos and word in wordfreq_dic_list[j]:
                a += wordfreq_dic_list[j][word]/total_freq[j]
        fp = int(total_freq[ignore_pos]*0.00001)
        coeff = wordfreq_dic_list[ignore_pos][word] - fp
        if coeff <= 1:
            coeff = 1
        num = (1 - 1/coeff)*((main_vocab_dic[word]/total_freq[ignore_pos])/a)
        holder[word] = num
    return holder


def run_slang_analyser(subnames, save_name='temporary', save_file=True):
    """
    Purpose:  
        This function is the main function that encapsulates the logic for all the steps required to run
        affinity analysis. The affinity analysis should extract key terms. 
    
    Parameters:
        subnames: a list of subreddit names, of which directories should be accessed and their lemma dictionaries
                should be used to extract high affinity terms, relative to other subreddits. -> list -> str
        target_sub:
                a subreddit that should be loaded, separetly,and its file should be loaded from disc. 
    """
    sub_dirs_path ='/home/ndg/projects/shared_datasets/semantic_shift_lemmatized/subreddits_nov2014_june2015/'
    
    # load every single word-frequency dictionary for each subname.
    
    wordfreq_ext = '_filtered_lemma.pkl'
    
    # list of wordfreq dictionary list. 
    wordfreq_dic_list = []
    
    num_of_processes = os.cpu_count() - 1
    pool = mp.Pool(processes=num_of_processes)
    
    # Loads the lemma_dictionaries for each subreddit in the slang_analyser function. 
    for sub in subnames:
        lemma_dic_path = sub + wordfreq_ext
        sub_lemma_dic_path = os.path.join(
            os.path.join(sub_dirs_path, sub), 
            lemma_dic_path)
        wordfreq_dic_list.append(pickle_load(sub_lemma_dic_path))
    
    print(len(wordfreq_dic_list))
    total_freq = calculate_total_freq(wordfreq_dic_list, mult=True)
    
    print(len(total_freq))
    
    affinity_dic_list = []
    
    for i_position in range(len(subnames)):
        affinity_dic_list.append(pool.apply(affinity_analysis, args=(wordfreq_dic_list, total_freq, i_position)))
    
    if save_file:
        affinity_path = os.path.join(sub_dirs_path, 'affinity_values')
        if not os.path.exists(affinity_path):
            os.makedirs(affinity_path)
        pickle_dump(save_name, '_affinity_terms.pkl', affinity_path, affinity_dic_list)
    else:
        return affinity_dic_list
    

if __name__ == '__main__':
    # Decide how the subnames are going to be parsed. 
    
    # maybe load it through a file or a an import function.
    subname_file = sys.argv[1]
#     subnames = pickle_load(subname_file)
    
    with open(subname_file) as f:
        subnames = f.read()
#     subnames = subnames.split('\n')
    print(type(subnames))
    subnames = subnames.split('\n')[:-1]
    print(len(subnames))
    run_slang_analyser(subnames)