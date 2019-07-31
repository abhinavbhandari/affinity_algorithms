import glob
from subreddinfo import pickle_load
import wordvectoranalysis
import sys
import os

# declaring it as a global variable DANGEROUS
save_dir='/home/ndg/projects/shared_datasets/semantic_shift_lemmatized/subreddits_nov2014_june2015/'

def get_full_text(all_files, freq):
    text = []
    for file in all_files:
        lemmas = pickle_load(file)
        text.extend(lemmas)
    filtered_text = wordvectoranalysis.ready_corpus_for_w2v(text, None, freq)
    return filtered_text


def make_model_and_corpus(word_freq, subreddit, years=['2014'], months=['11', '12'], load_data_from='/home/ndg/projects/shared_datasets/semantic_shift_lemmatized/'):
    
    sub_dir = os.path.join(load_data_from, subreddit)
    
    file_regexes = []
    
    for y in range(len(years)):
#         save_dir = save_dir + years[y]
        for m in range(len(months)):
            file_regex = '*' + years[y] + '-' + months[m] + '*'
            file_regexes.extend(glob.glob(os.path.join(sub_dir, file_regex)))

    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    
    lemma_text = get_full_text(file_regexes, word_freq)
    
    embedding_model = wordvectoranalysis.create_w2v_model(lemma_text)
    
    save_sub_path = os.path.join(save_dir, subreddit)
    save_file_path = os.path.join(save_sub_path, subreddit + '-' + years[0] + '-' + months[0] + '-' + months[1] + '.model')
    embedding_model.save(save_file_path)


if __name__ == "__main__":
    # load the semantic lemmatized's dataset. 
    
    years=[['2014'], ['2015'], ['2015'], ['2015']]
    months=[['11', '12'], ['01', '02'], ['03', '04'], ['05', '06']]
    
    subreddit = sys.argv[1]
    print(subreddit)
    load_path = os.path.join(save_dir, subreddit)
    dic_path = os.path.join(load_path, subreddit + '_filtered_lemma.pkl')
    filtered_lemma_dic = pickle_load(dic_path)
        
    for year, month in zip(years, months):
        make_model_and_corpus(filtered_lemma_dic, subreddit, year, month)