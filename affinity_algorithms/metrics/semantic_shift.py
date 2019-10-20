import gensim
from affinity_algorithms.preprocessing import lemmatize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
stopw = set(stopwords.words('english'))
import os
import sys
import numpy as np
import multiprocessing as mp
import pandas as pd

def filter_word_vectors(subnames,
                        aff_words,
                        default_path='/home/ndg/projects/semantic_shift_lemmatized/subreddits_nov2014_june2015/', 
                        model_ext=['-2014-11-12.model', '-2015-01-02.model', '-2015-03-04.model', '-2015-05-06.model']):
    """Filters word vectors that only have four model extensions.
    
    This is a filtering algorithm that only extracts word models for sizeably
    trained models. Often times large models in word2vec are distributed into
    multiple files. We ignore those models.
    
    Args:
        subnames (list): List of subreddits
        aff_words (list): List of terms
        default_path (str): Default path from which models should be extracted.
        model_ext (list): Model extension to observe
        
    Returns:
        new_subnames, new_aff_words
    """
    remove_index = []
    new_aff_words = []
    new_subnames = []
    for i, sub in enumerate(subnames):
        count = 0
        sub_dir = os.path.join(default_path, sub)
        for file_names in os.listdir(sub_dir):
            if '.model' in file_names:
                count += 1
        
        if count == 4:
            new_subnames.append(sub)
            new_aff_words.append(aff_words[i])
    return new_subnames, new_aff_words


def get_word_models(subreddit, 
                    default_path=None, 
                    model_ext=['-2014-11-12.model',
                               '-2015-01-02.model',
                               '-2015-03-04.model',
                               '-2015-05-06.model']):
    """Returns the word2vec models searched for in a subreddit directory. 
        
    By default, it returns in the intervals of 2 months, word2vec models 
    trained between Nov 2014 and June 2015. 
    
    The default path should be
    '/home/ndg/projects/semantic_shift_lemmatized/subreddits_nov2014_june2015/'.
        
    Args:
        subreddit (str): subreddit name
        default_path (str): path to the subreddit models directory
        model_ext (list): extension by which the models are loaded
    
    Returns:
        list of models
    """
    subreddit_dir = os.path.join(default_path, subreddit)
    
    models = []
    for exts in model_ext:
        models.append(Word2Vec.load(os.path.join(subreddit_dir, subreddit + exts)))
    return models


def ready_corpus_for_w2v(posts,
                         nlp,
                         freq=None,
                         batch_size=100,
                         thread_count=8,
                         lemmatize_bool=False,
                         filter_stopw=True):
    """Converts corpus for training a Word2Vec.
    
    Readying the corpus includes lemmatizing.
    
    Args:
        posts (list): List of text.
        nlp (spacy.nlp): Spacy parser
        freq (dic): Word frequencies
        batch_size (int): Default is 100
        thread_count (int): Default is 8
        filter_stopw (bool): Boolean for filtering
        
    Returns:
        A processed converted corpus.
    """
    words = set(freq.keys())
    
    convert_corpus = [[w for w in p if w in words and w not in stopw] for p in posts]
    
    if lemmatize_bool:
        texts = [' '.join(p) for p in convert_corpus]
        convert_corpus = lemmatize.lemmatize_text(texts, nlp, batch_size=batch_size, thread_count=thread_count)
    
    return convert_corpus


def create_w2v_model(convert_corpus):
    """Wrapper for word2vec model of a text corpus.
    
    Args:
        convert_corpus (list): List of textual corpus
        
    Returns:
        Word2Vec.Model
    """
    model = gensim.models.Word2Vec(
        convert_corpus,
        size=300,
        window=5,
        min_count=5,
        workers=10)
    
    model.train(convert_corpus, total_examples=len(convert_corpus), epochs=10)
    return model


def word_cross_comparison(similar_words):
    """Computes intersection of words from one list the other lists
    
    Args:
        similar_words (list -> list): nested list of neighboring words
        
    Returns:
        Score matrix of similarities
    """
    score_mat = [[] for i in range(len(similar_words))]
    for i, w_list in enumerate(similar_words):
        for j, s_list in enumerate(similar_words):
            count = 0
            for w in w_list:
                if w in s_list:
                    count += 1
            coeff_score = count/10
            score_mat[i].append(coeff_score)
    return score_mat


def get_similarity_score(pearson_mat):
    """Computes stability scores as similarity."""
    total_mat = pearson_mat[0]
    for mat in pearson_mat[1:]:
        total_mat += mat
    return total_mat/len(pearson_mat)


def compute_semantic_shift_matrix(word_embeddings, k=10, smoothing=0.1, jaccard=True):
    """Computes semantic shift matrix for a subreddit's word embeddings.
    
    Computes semantic shift matrix by comparing the k similar neighbors of
    word embeddings that are ordered.
    
    Args:
        word_embeddings (word2vec.Models): list of word embeddings ordered by date interval
        k (int): number of similar neighbors to compare
        smoothing (bool): smoothing coefficient, that gives an assumption of at least 
            one neighbor is similar (although the data may not contain it).
        jaccard (bool):
    
    Returns:
        word_embeddings
    """
    word_embeddings_len = len(word_embeddings)
    comp_mat = np.empty((word_embeddings_len, word_embeddings_len))
    for i, ws in enumerate(word_embeddings):
        for j, ws2 in enumerate(word_embeddings):
            count = 0
            for w in ws:
                if w in ws2:
                    count += 1
            if jaccard:
                f_score = count/(2*k - count) + smoothing
            else:
                f_score = count/k + smoothing
            if f_score == 1.1:
                f_score = 1
            comp_mat[i][j] = f_score
    return comp_mat


def semantic_shift_matrix(mods, aff_words, k=10, smoothing=0.1):
    """Extracts the semantic shift matrices for affinity terms by comparing 
        word embeddings.
    
    Args:
        mods (word2vec.Models): word embedding models for each interval, ordered 
            by date.
        aff_words (list): List of affinity terms
        k (int): Number of neighbors to compare
        smoothing (float): Laplace smoothing
    
    Returns:
       A list of semantic shift matrices, the number of words that are 
           in all four models.
    """
    word_to_score = {}
    pearson_mat = []
    word_count_in_models = 0
    w2a_indexes = []
    for w in aff_words:
        nextWord = False
        for m in mods:
            if w not in m.wv:
                nextWord = True
        if nextWord:
            word_to_score[w] = 0
            continue
        else:
            word_count_in_models += 1
        word_sets = []
        for m in mods:
            w_set = [t[0] for t in m.wv.similar_by_word(w)]
            word_sets.append(w_set)
        comp_mat = compute_semantic_shift_matrix(word_sets)
        word_to_score[w] = word_count_in_models
        pearson_mat.append(comp_mat)
        w2a_indexes.append(w)
    return pearson_mat, word_count_in_models, word_to_score, w2a_indexes


def get_forward_backward_scores(sim_matrix, net=True):
    """Extracts the semantic shift score of a subreddit.
    
    It does this in a few ways. Firstly from the similarity matrix, 
    it extracts the top left score as forward score.
    It then extracts the bottom right as backward score.
    
    Args:
        sim_matrix (ndarray): Matrix of similarity
        net (bool): Bool for computing net shift.
        
    Returns:
        list of semantic shifts scores.
    """
    net_forward_score = 0
    net_backward_score = 0
    forward_score = sim_matrix[0][1]
    backward_score = sim_matrix[-1][-2]
    if net:
        prev = 0
        for i in sim_matrix[0][1:]:
            if prev == 0:
                prev = i
            else:
                net_forward_score += i - prev
                prev = i
        prev = 0
        for j in sim_matrix[-1][:-1]:
            if prev == 0:
                prev = j
            else:
                net_backward_score += j - prev
                prev = j
    return [forward_score, backward_score, net_forward_score, net_backward_score]


def get_n_affinity_words_semantic_shift(subreddit, aff_words, net=True):
    """Top n affinity terms have their semantic shift evaluated.
    
    Conduct semantic shift analysis on a given set of words. The 
    given set of words are high affinity terms to a subreddit. 
    
    Args:
        subreddit: name of subreddit (str)
        aff_words: list of affinity terms. 
        
    Returns:
        semantic_scores, word_to_aff_score
    """
    models = get_word_models(subreddit)
    pearson_mat, word_count_in_mods, word_to_aff_score, w2a_indexes = semantic_shift_matrix(models, aff_words)
    try:
        for i, mat in enumerate(pearson_mat):
            word_to_aff_score[w2a_indexes[i]] = get_forward_backward_scores(mat)
        sim_matrix = get_similarity_score(pearson_mat)
        semantic_scores = get_forward_backward_scores(sim_matrix, net)
    except IndexError as e:
        semantic_scores = [0, 0, 0, 0]
    semantic_scores.append(word_count_in_mods)
    return semantic_scores, word_to_aff_score


def semantic_shift_analysis(subreddit, aff_words, net=True):
    """Conduct semantic shift analysis on a given set of words. 
    
    The given set of words are high affinity terms to a subreddit. 
        
    Args:
        subreddit (str): Subreddit name
        aff_words (list): Affinity terms for analysis
        net (bool): Boolean for calculating net semantic shift. Default is True.
        
    Returns:
        semantic_scores
    """
    models = get_word_models(subreddit)
    pearson_mat, word_count_in_mods, _, _ = semantic_shift_matrix(models, aff_words)
    try:
        sim_matrix = get_similarity_score(pearson_mat)
        semantic_scores = get_forward_backward_scores(sim_matrix, net)
    except IndexError as e:
        semantic_scores = [0, 0, 0, 0]
    semantic_scores.append(word_count_in_mods)
    return semantic_scores


def semantic_shift_analysis_mult(subreddits, aff_words, net=True):
    """Computes semamantic shift analysis using multiprocessing.
    
    Args:
        subreddits (list): Subreddit names
        aff_words (list): List of Affinity terms for measuing semantic shift
        net (bool): Boolean to measure net semantic shift. Default is True.
        
    Returns:
        semantic_scores_list, ordered by subreddit list.
    """
    semantic_scores_list = []
    
    num_of_processes = os.cpu_count() - 1
    pool = mp.Pool(processes=num_of_processes)
    for sub, aff_w in zip(subreddits, aff_words):
        semantic_scores_list.append(pool.apply(semantic_shift_analysis, args=(sub, aff_w)))
    return semantic_scores_list


def generate_subreddit_similarity_heatmap(multiscores, dates, subnames, mult=True):
    """Creates a heatmap that maps out the semantic shift in scores over intervals.
    
    Takes in semantic shift scores across different subreddits, then creates a heatmap
    which demonstrates the semantic shift for each subreddit across the specified time intervals.
    
    Args:
        multiscores: a list of semantic shift matrices
        dates: dates for time intervals (should be the same number as the dimension 
            of multiescore matrix)
        subnames: a list of subreddits.
        
    Returns:
        A dataframe
    """
    if mult:
        df = None
        for i in range(len(multiscores)):
            mc = multiscores[i]
            sub = subnames[i]
            sub_index = [sub] * len(mc)
            some_dataframe = pd.DataFrame(data=mc, index=[sub_index, dates], columns=dates)
            if type(df) == type(None):
                df = some_dataframe
            else:
                
                df = df.append(some_dataframe)
    return df.T

