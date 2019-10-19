import json

import sys
import os
import re
import pickle
import config
import glob


def extract_lemmas(full_path, lemma_dict, save_path="/home/ndg/projects/shared_datasets/semantic_shift_lemmatized/"):
    
    text_lines = pickle_load(full_path)
    for line in text_lines:
        for w in line:
            if w in lemma_dict:
                lemma_dict[w] += 1
            else:
                lemma_dict[w] = 1


def extract_lemma_dics_time_intervals(subname, time_intervals=['RC_2014-[1][12]*', 
                                                               'RC_2015-0[12]*', 
                                                               'RC_2015-0[34]*', 
                                                               'RC_2015-0[56]*']):
    sub_path = os.path.join(config.PROJECT_DATA_PATH, subname)
    save_path = os.path.join(config.SUBDIR_ANALYSIS_LOAD_PATH, subname)
    for ti in time_intervals:
        lemma_dict = {}
        ti_path = os.path.join(sub_path, ti)
        for full_path in glob.glob(ti_path):
            extract_lemmas(full_path, lemma_dict)
        lemma_save_file = subname + '_' + ti + '.pkl'
        lemma_save_path = os.path.join(save_path, lemma_save_file)
        pickle_dump(lemma_save_path, lemma_dict)


if __name__ == "__main__":
    # loads semantic_lemma's lemmatized files path. 
    
    lemma_dict = {}
    # should only load or pipe from file. 
    subreddit_name = sys.argv[1]
    
    extract_lemma_dics_time_intervals(subreddit_name)