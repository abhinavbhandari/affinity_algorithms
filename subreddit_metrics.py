import config
import slang_analyser
import wordvectoranalysis
import wordpair
import user_metrics
from subreddinfo import pickle_load, pickle_dump
import os
import sys
import pandas as pd
import numpy as np

def get_subreddit_metrics(sub, metrics_obj):
    loyalty = metrics_obj.intersect_loyalty[sub]
    dedication = metrics_obj.intersect_dedicated[sub]
    users = metrics_obj.intersect_users[sub]
    comments = metrics_obj.intersect_comments[sub]
    return [loyalty, dedication, users, comments]

class AffinityMetrics(object):
    def __init__(self, subnames, subreddit_metrics=None):
        """
        Extracts the informatino regarding n most afffnity words, their semantic shift, and their username metrics.
        """
        
        if not subreddit_metrics:
            subreddit_metrics = pickle_load(os.path.join(config.SUBREDDIT_METRIC, 
                                                         config.METRICS_EXT)
                                           )
        
        # convert subreddits to a dictionary where subreddit_name maps to index.
        self.sub_index_dic = {s_name: i for i, s_name in enumerate(subnames)}
        
        
        self.affinity_terms = slang_analyser.run_slang_analyser(subnames, save_file=False)
        n_affinity_terms = slang_analyser.get_n_most_affinity_terms_mult(self.affinity_terms)


    #     avg_aff_val = np.mean([affinity_terms[aff] for n_sub in n_affinity_terms])
        self.new_subnames, self.new_aff_words = wordvectoranalysis.filter_word_vectors(subnames, n_affinity_terms)
        sample_results = wordvectoranalysis.semantic_shift_analysis_mult(self.new_subnames, self.new_aff_words)
        sample_metrics = wordpair.calculate_user_name_metrics_mult(self.new_subnames, self.new_aff_words, subreddit_metrics.intersect_users)
        
        
        # This is a really confusing function, gotta clean it up. 
        avg_aff_val = []
        for n, sub_n in enumerate(self.new_aff_words):
            sample_metrics[n].append(np.mean([self.affinity_terms[self.sub_index_dic[self.new_subnames[n]]][aff] 
                                              for aff in sub_n]))

        metrics_data = [get_subreddit_metrics(sub, subreddit_metrics) for sub in self.new_subnames]
        
        
        self.metrics_df = pd.DataFrame(sample_metrics, 
                 index=new_subnames, 
                 columns=['mean', 'std', 'slang_to_user_wc',  'user_percent', 'average_affinity_value'])

        self.results_df = pd.DataFrame(sample_results, 
                 index=new_subnames, 
                 columns=['forward_shift', 'backward_shift', 'net_forward_shift', 'net_backward_shift', 'aff_wc_in_models'])


        self.subreddit_metrics_df = pd.DataFrame(metrics_data, 
                                            index=new_subnames, 
                                            columns=['loyalty', 'dedication', 'no._of_users', 'no._of_comments'])

        self.affinity_metrics_df = pd.concat([self.metrics_df, self.results_df, self.subreddit_metrics_df], axis=1)
        
    
        
#         return self.affinity_metrics_df
        

# def generate_subreddit_metrics_df(subnames, subreddit_metrics=None):
#     """
#     Extracts the informatino regarding n most afffnity words, their semantic shift, and their username metrics.
#     """
#     if not subreddit_metrics:
#         subreddit_metrics = pickle_load(os.path.join(config.SUBREDDIT_METRIC, 
#                                                      config.METRICS_EXT)
#                                        )
    
#     affinity_terms = slang_analyser.run_slang_analyser(subnames, save_file=False)
#     n_affinity_terms = slang_analyser.get_n_most_affinity_terms_mult(affinity_terms)
    
    
# #     avg_aff_val = np.mean([affinity_terms[aff] for n_sub in n_affinity_terms])
#     new_subnames, new_sample_aff_words = wordvectoranalysis.filter_word_vectors(subnames, n_affinity_terms)
#     sample_results = wordvectoranalysis.semantic_shift_analysis_mult(new_subnames, new_sample_aff_words)
#     sample_metrics = wordpair.calculate_user_name_metrics_mult(new_subnames, new_sample_aff_words, subreddit_metrics.intersect_users)
    
#     avg_aff_val = []
#     for n, sub_n in enumerate(n_affinity_terms):
#         sample_metrics[n].append(np.mean([affinity_terms[n][aff] for aff in sub_n]))
    
#     metrics_data = [get_subreddit_metrics(sub, subreddit_metrics) for sub in new_subnames]
    
    
#     # Change the way average affinity value is added
# #     print(type(avg_aff_val))
# #     sample_metrics.extend(avg_aff_val)
#     metrics_df = pd.DataFrame(sample_metrics, 
#              index=new_subnames, 
#              columns=['mean', 'std', 'slang_to_user_wc',  'user_percent', 'average_affinity_value'])
    
#     results_df = pd.DataFrame(sample_results, 
#              index=new_subnames, 
#              columns=['forward_shift', 'backward_shift', 'net_forward_shift', 'net_backward_shift'])
    
    
#     subreddit_metrics_df = pd.DataFrame(metrics_data, 
#                                         index=new_subnames, 
#                                         columns=['loyalty', 'dedication', 'no._of_users', 'no._of_comments'])
    
#     final_df = pd.concat([metrics_df, results_df, subreddit_metrics_df], axis=1)
    
#     return final_df
    