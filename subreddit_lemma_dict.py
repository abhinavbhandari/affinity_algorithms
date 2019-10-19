import json

import sys
import os
import re
# import pandas as pd
import pickle
import config
import glob


def extract_lemmas(base_path, partial_path, lemma_dict, save_path="/home/ndg/projects/shared_datasets/semantic_shift_lemmatized/"):
    
    full_path = base_path + '/' + partial_path
    subreddit_name = base_path.split('/')[-1]
    
    text_lines = pickle_load(full_path)
    for line in text_lines:
        for w in line:
            if w in lemma_dict:
                lemma_dict[w] += 1
            else:
                lemma_dict[w] = 1


if __name__ == "__main__":
    # loads semantic_lemma's lemmatized files path. 
    
    lemma_dict = {}
    # should only load or pipe from file. 
    subreddit_name = sys.argv[1]
    
    lemma_files_load_path = os.path.join(config.PROJECT_DATA_PATH, subreddit_name)
    for files in os.listdir(lemma_files_load_path):
        if 'lemmatized' not in files:
            continue
        file_path = os.path.join(lemma_files_load_path, files)
        base_path = "/".join(file_path.split("/")[:-1])
        partial_path = file_path.split("/")[-1]
        extract_lemmas(base_path, partial_path, lemma_dict)

    # Creates the subreddit directory in the parent save location. Only if it doesn't exist. 
    sub_save_dir = os.path.join(config.SUBDIR_ANALYSIS_LOAD_PATH, subreddit_name)

    save_file_name = subreddit_name + '_lemma_dic.pkl'

    full_save_file_name = os.path.join(sub_save_dir, save_file_name)
    
    
    if not os.path.exists(sub_save_dir):
        os.makedirs(sub_save_dir)
    print(full_save_file_name)
    pickle_dump(full_save_file_name, lemma_dict)