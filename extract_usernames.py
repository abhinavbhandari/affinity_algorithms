import wordpair
from subreddinfo import pickle_load, pickle_dump

# import numpy as np
import json

import sys
import os
# import pandas as pd
import pickle

# def pickle_dump(sub, ext, path, data):
#     fname = sub + ext
#     with open(os.path.join(path, fname), 'wb') as f:
#         pickle.dump(data, f)
#     f.close()


def get_usernames(file_name, unames):
    
    f_pointer = open(file_name)
    
    for line in f_pointer:
        try:
            sub_json = json.loads(line)
        except:
            print(line, file_name)
        
        if sub_json['author'] not in unames:
            unames.add(sub_json['author'])
        
    

if __name__ == "__main__":
    # load caitrin's extracted jsons to extract desired users. 
    
    lemma_file_path = '/home/ndg/projects/shared_datasets/semantic_shift_lemmatized/'
    default_save_dir = '/home/ndg/projects/shared_datasets/semantic_shift_lemmatized/subreddits_nov2014_june2015/'
    dict_file_ext = '_lemma_dic.pkl'
    
    subreddit = sys.argv[1].split('/')[-2]
    
    print(subreddit)
    
    uname_dl = False
    uname_path = subreddit + '_usernames.pkl'
    uname_abs_path = os.path.join(os.path.join(default_save_dir, subreddit), uname_path)
    if not os.path.exists(uname_path): 
        uname_dl = True
    
    # extracts usernames for subreddit
    if uname_dl:
        unames = set()
    for file_name in os.listdir(sys.argv[1]):
        abs_file_name = os.path.join(sys.argv[1], file_name)
        if uname_dl:
            get_usernames(abs_file_name, unames)
    
    # calculate the path where the lemma dic is stored. 
    sub_dir_file_path = os.path.join(default_save_dir, subreddit)
    dict_file_path = os.path.join(sub_dir_file_path, (subreddit + dict_file_ext))
    
    # loads the lemma dic from subreddit's storage folder. 
    lemma_dict = pickle_load(dict_file_path)
    
    # conduct subreddit analysis. 
    lemma_tups = wordpair.dic_to_tup(lemma_dict)
    max_len = max(len(k) for k in unames)
    filtered_lemma_tups, removed_lemma_tups = wordpair.filter_words([lemma_tups], max_len=max_len)
    filtered_lemma_tups = filtered_lemma_tups[0]
    removed_lemma_tups = removed_lemma_tups[0]
    filtered_lemma_dic = wordpair.tup_to_dic(filtered_lemma_tups)
    w2u, u2w = wordpair.create_user_to_word(unames, filtered_lemma_dic)
    wordpair.remove_words_not_in_dic(filtered_lemma_dic, w2u, u2w)
    w2u, u2w = wordpair.filter_correct_word_and_user(w2u, u2w)
    
    
    save_dir = os.path.join(default_save_dir, subreddit)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    pickle_dump(subreddit, '_filtered_lemma.pkl', save_dir, filtered_lemma_dic)
    pickle_dump(subreddit, '_w2u.pkl', save_dir, w2u)
    pickle_dump(subreddit, '_u2w.pkl', save_dir, u2w)
    
    if uname_dl:
        pickle_dump(subreddit, '_usernames.pkl', save_dir, unames)
        