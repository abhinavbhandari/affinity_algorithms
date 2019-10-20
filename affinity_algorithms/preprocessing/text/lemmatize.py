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
    """Process text and remove filler characters.
    
    Modifies basic terms such as "it's", "can't", "n't".
    
    Args:
        text (str): text
        
    Returns:
        processed text
    """
    text = re.sub(r"it\'s", "it is", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r"[^A-Za-z]", " ", text)
    return text


def lemmatize_text(preprocessed_text,
                   batch_size=100,
                   thread_count=8):
    """Accepts preprocessed_text and then lemmatizes them.
    
    NLP lemmatizer is preloaded and this function uses nlp pipes for optimized extraction.
    After doing cross validation of parameters on computer settings, the default
    batch size and thread counts have been set.
    
    Args:
        preprocessed_text (list):  List of words
        batch_size (int): Batches in which to lemmatize
        thread_count (int): number of threads to create
        
    Returns:
        List of tokens
    """
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


def task_handling(base_path, 
                  partial_path, 
                  save_path="/home/ndg/projects/shared_datasets/semantic_shift_lemmatized/", 
                  unames=None):
    """Task handling takes in the path from which it should load text.
    
    Task handler loads files for analysis from provided path. It then processes
    the text and continues to lemmatize it. After lemmatizing the text, it saves it
    into the provided subreddit directory.
    
    Args:
        base_path (str): Directory path
        partial_path (str): Relative path
        save_path (str): Path to save files
        unames (set): Set of usernames
    """
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
    """Separate function used for lemmatizing directory files.
    
    A path is provided, it loads files from that path and calls the task handler.
    
    Args:
        dir_path (str): Directory path
        subreddit (str): target subreddit
        
    Saves:
        dumps a pickle of lemmatized tokens in the subreddit directory.
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
    """A subreddit is passed and its textual data is then evaluted for lemmas.
    """
    dir_path = os.path.join(config.SUBREDDIT_STORAGE_PATH, subreddit)
    lemmatize_text_files(dir_path, subreddit)
    

if __name__ == "__main__":
    """Main method is invoked for when running with GNU parallel through cmdline.
    
    Args:
        sys.argv[1]: subreddit name
    """
    subreddit = sys.argv[1]
    lemmatize_text_files_from_full_path(subreddit)
