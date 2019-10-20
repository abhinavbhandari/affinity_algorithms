import config
from affinity_algorithms.metrics import loyalty, semantic_shift, process_usernames, community, affinity
from affinity_algorithms.utils import pickle_load, pickle_dump
import os
import sys
import pandas as pd
import numpy as np

#--------------------------
# This module creates the dataframes 
# that encapsulate metrics for each 
# subreddit. These metrics include
# semantic shift, affinity analysis,
# username retrieval and community
# measurements.
# 
# Global Variables
#--------------------------
sub_index_dic = None
affinity_terms = None
new_subnames = None
new_sample_aff_words = None


def get_subreddit_attributes(sub, metrics_obj):
    """Return loyalty, dedication, users and comments by extracting from a subreddit metrics object."""
    loyalty = metrics_obj.intersect_loyalty[sub]
    dedication = metrics_obj.intersect_dedicated[sub]
    users = metrics_obj.intersect_users[sub]
    comments = metrics_obj.intersect_comments[sub]
    return [loyalty, dedication, users, comments]


def get_subreddit_metrics(new_subnames, subreddit_metrics_local):
    """ Return a dataframe with subreddit metrics for each subreddit. """
    global subreddit_metrics
    if not subreddit_metrics_local:
        subreddit_metrics = pickle_load(os.path.join(config.SUBREDDIT_METRIC, 
                                                     config.METRICS_EXT)
                                       )
    else:
        subreddit_metrics = subreddit_metrics_local
    
    subreddit_metrics_data = [get_subreddit_attributes(sub, subreddit_metrics) 
                              for sub in new_subnames]
    
    subreddit_metrics_df = pd.DataFrame(subreddit_metrics_data,
                                        index=new_subnames,
                                        columns=['loyalty', 
                                                 'dedication',
                                                 'no._of_users',
                                                 'no._of_comments'])
    return subreddit_metrics_df


def load_semantic_shift_vectors(new_subnames,
                                new_sample_aff_words,
                                n=100,
                                load_style="full_load",
                                aff_ext="_neutral_1000.pkl"):
    """Retrieves the semantic shift vectors for each subreddit.
    
    The semantic shift vectors include foward stability, backward stability,
    forward net shift, and backward net shift. These measurements are then added onto
    the main dataframe.
    
    Args:
        new_subnames (list): List of subnames
        new_sample_aff_words (list): list of affinity terms
        n (int): Number of terms to compute semantic shift on
        load_style (str): style of load
        aff_ext (str): File extension for when saving a file
        
    Returns:
        list of semantic vectors for each term. 
    """
    aff_list = []
    for sub, aff_words in zip(new_subnames, new_sample_aff_words):
        sub_space = os.path.join(config.SUBDIR_ANALYSIS_LOAD_PATH, sub)
        file_name = sub + aff_ext
        full_path = os.path.join(sub_space, file_name)
        word_to_aff_dic = pickle_load(full_path)
        if load_style == "full_load":
            semantic_shift_list = [word_to_aff_dic[i] for i in aff_words if word_to_aff_dic[i] != 0]
            semantic_shift_len = len(semantic_shift_list)
            try:
                semantic_shift_ave = list(np.mean(semantic_shift_list, axis=0))
                semantic_shift_ave.append(semantic_shift_len)
            except:
                semantic_shift_ave=[0, 0, 0, 0, 0]
            semantic_shift_ave.append(0)
            aff_list.append(semantic_shift_ave)
        elif load_style == "half_load":
            aff_dic = pickle_load(os.path.join(sub_space, sub + "_affinity_terms.pkl"))
            count = 0
            sub_aff = []
            aff_values = []
            for w in aff_words:
                if count == n:
                    break
                elif word_to_aff_dic[w] != 0:
                    sub_aff.append(word_to_aff_dic[w])
                    aff_values.append(aff_dic[w])
                    count+=1
            try:
                semantic_shift_ave = list(np.mean(sub_aff, axis=0))
                semantic_shift_ave.append(count)
                semantic_shift_ave.append(np.mean(aff_values))
            except:
                semantic_shift_ave=[0, 0, 0, 0, 0, 0]
            aff_list.append(semantic_shift_ave)
    return aff_list


def get_semantic_shift_metrics(new_subnames, new_sample_aff_words, n=100, j=100, load_style="full_load"):
    """Computes the semantic shift for affinity top n affinity terms
    
    Takes in a list of subreddits and their top n affinity terms, and computes the semantic shift for each term
    across time intervals.
    
    Args:
        new_subnames: list of subreddits
        new_sample_aff_words: list of (list of top n affinity terms for a subreddit)
    
    Returns:
        
    """
    if load_style == "create":
        semantic_shift_metrics = semantic_shift.semantic_shift_analysis_mult(new_subnames, new_sample_aff_words)
    else:
        semantic_shift_metrics = load_semantic_shift_vectors(new_subnames, new_sample_aff_words, j, load_style)
    
    results_df = pd.DataFrame(semantic_shift_metrics, 
             index=new_subnames, 
             columns=['forward_shift',
                      'backward_shift',
                      'net_forward_shift',
                      'net_backward_shift',
                      'aff_wc_in_models',
                      'half_load_affinity_average'])
    
    return results_df


def get_user_metrics(new_subnames, new_sample_aff_words):
    """Computes username metrics for the subreddits that are passed.
    
    Username metrics in relation to subreddit affinity terms is the number of affinity terms
    that are adopted by users, and to what extent. This is calculated by metrics such as the 
    mean number of users that adopt high affinity terms, standard deviation, number of slang 
    terms adopted by users, user-percent and average affinity value.
    
    Args:
        new_subnames: list of subreddits
        new_sample_aff_words: list of (list of top n affinity terms for a subreddit)
        
    Returns:
        Returns a dataframe which consists the mean, std, slang-to-user-word-count, 
        percentage of users adopting slang terms and the average affinity values of top n terms
    """
    
    username_metrics = process_usernames.calculate_user_name_metrics_mult(new_subnames,
                                                                 new_sample_aff_words,
                                                                 subreddit_metrics.intersect_users)
    
    avg_aff_val = []
    for n, sub_n in enumerate(new_sample_aff_words):
        username_metrics[n].append(np.mean([affinity_terms[sub_index_dic[new_subnames[n]]][aff] 
                                          for aff in sub_n]))
    
    username_metrics_df = pd.DataFrame(username_metrics, 
                 index=new_subnames, 
             columns=['mean','std','slang_to_user_wc','user_percent','average_affinity_value'])
    
    return username_metrics_df
    

def generate_subreddit_metrics_df(subnames,
                                  n=100,
                                  j=100,
                                  affinity_load_type="full",
                                  affinity_terms_path="default",
                                  load_style="full",
                                  subreddit_metrics=None):
    """Extracts the information regarding n most afffnity words, their semantic shift, 
        and their username metrics.
    
    Calculates the n most affinity words, semantic shift and how many users have adopt 
    affinity terms for each subreddit that is passed. 
    
    Args:
        subnames: list of subreddits
        n: number of top affinity terms that should be extracted and semantic shift should be measured.
        subreddit_metrics: An object that contains subreddit metric data for each subreddit such as 
                loyalty, dedication, no. of users and comments. If None, then it loads a default one.
                
    Returns:
        A final dataframe which includes affinity term, semantic shift and username data for each subreddit.
        Also a dictionary which is the affinity terms for each subreddit, and
    """
    
    global sub_index_dic
    sub_index_dic = {s_name: i for i, s_name in enumerate(subnames)}
    
    global affinity_terms
    if affinity_load_type == "full":
        if affinity_terms_path == "default":
            affinity_terms_path = os.path.join(config.SUBREDDIT_METRIC, 'all_subreddits_affinity_terms.pkl')
            affinity_terms = pickle_load(affinity_terms_path)
    elif affinity_load_type == "subs":
        affinity_terms = affinity.load_affinity_terms_mult(subnames)
    elif affinity_load_type == "create":
        affinity_terms = affinity.run_slang_analyser(subnames)
    
    n_affinity_terms = affinity.get_n_most_affinity_terms_mult(affinity_terms, n=n)
    
    global new_subnames, new_sample_aff_words
    new_subnames, new_sample_aff_words = semantic_shift.filter_word_vectors(subnames, n_affinity_terms)
    
    subreddit_metrics_df = get_subreddit_metrics(new_subnames, subreddit_metrics)
    semantic_shift_df = get_semantic_shift_metrics(new_subnames, new_sample_aff_words, j=j, load_style=load_style)
    username_metrics_df = get_user_metrics(new_subnames, new_sample_aff_words)
    
    final_df = pd.concat([username_metrics_df, semantic_shift_df, subreddit_metrics_df], axis=1)
    
    return final_df
    