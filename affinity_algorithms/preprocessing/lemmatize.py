# import numpy as np
import json
import nltk
from nltk import sent_tokenize
import spacy
from affinity_aglorithms.utils.files import pickle_load, pickle_dump
import sys
import os
import re
import pandas as pd
import pickle
import config

nlp = spacy.load('en', disable=['parser', 'ner'])

def preprocessing_text(text):
    text = re.sub(r"it\'s", "it is", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r"[^A-Za-z]", " ", text)
    return text


def lemmatize_text(preprocessed_text, batch_size=100, thread_count=8):
    token_list = []
    piped_docs = nlp.pipe(preprocessed_text, batch_size=batch_size, n_threads=thread_count)
    for doc in piped_docs:
        store_word = ''
        tokens = []
        for token in doc:
            if token.lemma_ != '-PRON-':
                store_word = token.lemma_
            else:
                store_word = token.lower_
            if store_word.strip() == '':
                continue
            else:
                tokens.append(store_word)
        token_list.append(tokens)
    return token_list


def task_handling(base_path, partial_path, save_path="/home/ndg/projects/shared_datasets/semantic_shift_lemmatized/", unames=None):
    # use pandas read_csv 
    full_path = base_path + '/' + partial_path
    subreddit_name = base_path.split('/')[-1]
    
    f_pointer = open(full_path)
    preprocessed_text = []
    for line in f_pointer:
        try:
            sub_json = json.loads(line)
        except:
            print(sub_name, line)
        
        if sub_json['author'] not in unames:
            unames.add(sub_json['author'])
        
        lemmatized_body = preprocessing_text(sub_json['body'])
        preprocessed_text.append(lemmatized_body)
        
    # check if its the keys. or use .values
    # lemmatized text
    try:
        lemmatized_text = lemmatize_text(preprocessed_text)
    except:
        print('Failed Lemmatization')
        raise Exception()
    
    sub_output_dir = save_path + subreddit_name + "/"
    if not os.path.exists(sub_output_dir):
        os.makedirs(sub_output_dir)
    
    ext = partial_path.split('.')[0] + '_lemmatized.pkl'
    full_save_path = sub_output_dir + '/' + ext
    with open(full_save_path, 'wb') as f:
        pickle.dump(lemmatized_text, f)


def lemmatize_text_files(dir_path, subreddit):
    """
    """
    unames = set()
    
    for files in os.listdir(dir_path):
        file_path = os.path.join(dir_path, files)
        base_path = "/".join(file_path.split("/")[:-1])
        partial_path = file_path.split("/")[-1]
        task_handling(base_path, partial_path, unames=unames)
        
    save_dir = os.path.join(config.SUBDIR_ANALYSIS_LOAD_PATH, subreddit)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    pickle_dump(subreddit, '_usernames.pkl', save_dir, unames)
    

def lemmatize_text_files_from_full_path(subreddit):
    dir_path = os.path.join(config.SUBREDDIT_STORAGE_PATH, subreddit)
    lemmatize_text_files(dir_path, subreddit)
    

if __name__ == "__main__":
    """
    """
    subreddit = sys.argv[1]
    lemmatize_text_files_from_full_path(subreddit)
