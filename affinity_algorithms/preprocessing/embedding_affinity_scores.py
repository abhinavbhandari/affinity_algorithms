import wordvectoranalysis
import config
from subreddinfo import pickle_load, pickle_dump
import os
import sys


def run_embedding_affinity_scores(subreddit, affinity_terms_path="default", n=1000, reverse=False):
    sub_space = os.path.join(config.SUBDIR_ANALYSIS_LOAD_PATH, subreddit)
    if affinity_terms_path == "default":
        aff_path = subreddit + "_affinity_terms.pkl"
        affinity_terms_path = os.path.join(sub_space, aff_path)
    affinity_terms = pickle_load(affinity_terms_path)
    aff_words = sorted(affinity_terms, key=lambda x: affinity_terms[x], reverse=reverse)
    if reverse:
        aff_1000 = aff_words[:1000]
    else:
        aff_words = [aff for aff in aff_words if affinity_terms[aff] != 0]
        aff_1000 = aff_words[:1000]
        
    _, word_to_aff_score = wordvectoranalysis.get_n_affinity_words_semantic_shift(subreddit, aff_1000)
    if reverse:
        save_str = '_aff2_' + str(n) + '.pkl'
    else:
        save_str = '_neutral2_' + str(n) + '.pkl'
    pickle_dump(subreddit, save_str, sub_space, word_to_aff_score)
    print(subreddit)
#     return semantic_scores, word_to_aff_score

if __name__ == '__main__':
    subreddit = sys.argv[1]
    run_embedding_affinity_scores(subreddit)
    
