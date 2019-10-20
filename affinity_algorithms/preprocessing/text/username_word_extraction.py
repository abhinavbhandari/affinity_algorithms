from affinity_algorithms.metrics import process_usernames
from affinity_aglorithms.utils.files import pickle_load, pickle_dump
from subreddinfo import pickle_load, pickle_dump
import json
import sys
import os
import pickle
import config


def get_usernames(file_name, unames):
    """Extracts usernames from passed path.
    
    Adds the extracted usernames to the unames set.
    
    Args:
        file_name (str): Path to file of data
        unames (set): Set of user data
    """
    f_pointer = open(file_name)
    
    for line in f_pointer:
        try:
            sub_json = json.loads(line)
        except:
            print(line, file_name)
        
        if sub_json['author'] not in unames:
            unames.add(sub_json['author'])


def load_usernames(subreddit):
    """Extracts usernames for a subreddit.
    
    The code is optimised to process usernames if they have already
    been extracted in storage directory.
    
    Args:
        subreddit (str): Name of subreddit
        
    Returns:
        unames (set): Set of usernames in the subreddit.
    """
    uname_dl = False
    uname_path = subreddit + '_usernames.pkl'
    sub_save_dir = os.path.join(config.SUBDIR_ANALYSIS_LOAD_PATH, subreddit)
    uname_abs_path = os.path.join(sub_save_dir, uname_path)
    if not os.path.exists(uname_abs_path): 
        uname_dl = True
        unames = set()
    else:
        print(subreddit)
        unames = pickle_load(uname_abs_path)
    
    # extracts usernames for subreddit
    if uname_dl:
        for file_name in os.listdir(sys.argv[1]):
            abs_file_name = os.path.join(sys.argv[1], file_name)
            get_usernames(abs_file_name, unames)


def load_subreddit_dictionary(subreddit):
    """Loads the lemmatized dictionary for a subreddit.
    
    This dictionary contains information about a's and b's
    
    Args:
        subreddit (str): Subreddit for which the dictionary 
            should be loaded.
    
    Returns:
        lemma_dict (dic): A dictionary with words and frequencies.
    """
    # calculate the path where the lemma dic is stored.
    dict_file_ext = '_lemma_dic.pkl'
    sub_dir_file_path = os.path.join(config.SUBDIR_ANALYSIS_LOAD_PATH, subreddit)
    dict_file_path = os.path.join(sub_dir_file_path, (subreddit + dict_file_ext))
    
    # loads the lemma dic from subreddit's storage folder. 
    lemma_dict = pickle_load(dict_file_path)
    return lemma_dict


def compute_w2u_and_u2w(unames, lemma_dict):
    """Computes the word2users, and user2word dictionaries.
    
    The function first identifies matching words from lemma_dict, then it 
    extracts these terms from each username in unames. This data is compiled
    into a new dictionary. There are many repetition cases where the words 
    identified in usernames are words within other words. For example, if "whenever" 
    is found in a username, "when" will also be found.
    
    Args:
        unames (set): Set of usernames
        lemma_dict (dic): Dictionary of lemmas -> frequency
        
    Returns:
        w2u, u2w
    """
    lemma_tups = process_usernames.dic_to_tup(lemma_dict)
    max_len = max(len(k) for k in unames)
    filtered_lemma_tups, removed_lemma_tups = process_usernames.filter_words([lemma_tups], max_len=max_len)
    filtered_lemma_tups = filtered_lemma_tups[0]
    removed_lemma_tups = removed_lemma_tups[0]
    filtered_lemma_dic = process_usernames.tup_to_dic(filtered_lemma_tups)
    w2u, u2w = process_usernames.create_user_to_word(unames, filtered_lemma_dic)
    process_usernames.remove_words_not_in_dic(filtered_lemma_dic, w2u, u2w)
    w2u, u2w = process_usernames.filter_correct_word_and_user(w2u, u2w)
    return w2u, u2w


def process_words_in_usernames(subreddit):
    """Takes in a subreddit and processes w2u, u2w extraction."""
    subreddit = sys.argv[1]
    
    unames = load_usernames(subreddit)
    lemma_dict = load_subreddit_dictionary(subreddit)
    w2u, u2w = compute_w2u_and_u2w(unames, lemma_dict)
    
    if not os.path.exists(sub_save_dir):
        os.makedirs(sub_save_dir)
    
    pickle_dump(subreddit, '_filtered_lemma.pkl', sub_save_dir, filtered_lemma_dic)
    pickle_dump(subreddit, '_w2u.pkl', sub_save_dir, w2u)
    pickle_dump(subreddit, '_u2w.pkl', sub_save_dir, u2w)
    

if __name__ == "__main__":
    subreddit = sys.argv[1]
    process_words_in_usernames(subreddit)