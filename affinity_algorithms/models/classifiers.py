import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


def calculate_f1_scores(y_test, y_pred, average=None):
    """Wrapper function for calculating F1 Score.
    """
    return f1_score(y_test, y_pred, average=average)


def combine_corps(sub_files, date, def_path='/home/ndg/users/abhand1/subreddit_data/models/'):
    """Loads Corpus and combines them across multiple dates.
    """
    
    main_corp = []
    
    for sub in sub_files:
        file_path = sub + '_' + date + '_corp.pkl'
        f_path = def_path + file_path
        main_corp.append(pickle_load(f_path))
        
    return main_corp


def make_predictions(clf,
                     x_tr, 
                     y_tr, 
                     x_te):
    """Takes a classifier, trains it and applies it onto the X test.
    
    Args:
        clf: Classifier
        x_tr: X Train Set
        y_tr: Y Train Set
        x_te: X Test Set
    
    Returns:
        None
    """
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)

    
def countvectorizing(stopwords=True, 
                     min_df=1, 
                     ng=(1,1)):
    """A wrapper function for initializing count vectorizer
    
    Args:
        stopwords: boolean value, whether to remove it or not.
        min_df: 1
        ng: Default is n-grams of (1,1).
        
    Returns:
        count_vect
    """
    if(stopwords):
        count_vect = CountVectorizer(ngram_range=ng, stop_words="english", min_df=min_df)
    else:
        count_vect = CountVectorizer(ngram_range=ng, min_df=min_df)
    return count_vect


def find_optimal_params(clf, 
                        corpus, 
                        y_vec, 
                        params=None, 
                        stopword_list=[True, False], 
                        pass_=True, 
                        final_count_vect=None, 
                        lb=None):
    """User parameter testing to extract the best model for classification.
    
    
    Args:
        clf: An Sklearn classifier
        corpus: List of textual data
        y_vec: Vector index for y values
        params: Default is none. None triggers cross validation
        stopword_list: A list of booleans that tell it to include it or not.
        pass_: boolean
        final_count_vect: Default is None
        lb: None
        
    Returns:
        best_clf model, validation matrix, results, final_count_vect, lb
        
    """
    #the list of ngrams tested
    ng_list = [(1,1), (1,2), (2,2)]
    #to include stopwords or not:
    
    #best clf from grid search and its accuracy score, currently stored with dummy values.
    best_clf = 0
    best_score = [0, 0, 0, 0]
    conf_matrix = []
    
    #best settings for countvectorizer which are initialized with dummy values
    stopw = False
    ng_val = (1,1)
    results = []
    if not lb:
        lb = LabelEncoder()
    for ngram in ng_list:
        for stopword in stopword_list:
            if not final_count_vect:
                count_vect = countvectorizing(ng=ngram, stopwords=stopword)
                x_value = count_vect.fit_transform(corpus)
                y_value = lb.fit_transform(y_vec)
            else:
                count_vect = final_count_vect
                x_value = count_vect.transform(corpus)
                y_value = lb.transform(y_vec)

            X_train, X_test, y_train, y_test = train_test_split(x_value, y_value, test_size=0.2, random_state=33)
            
            #runs grid search on the passed classifier with its range of parameters
            if (params!=None):
                clf_grid = GridSearchCV(clf, params, cv=5)
            else:
                #just reset it as the baseline model.
                clf_grid = clf
            if pass_:
                clf_grid.fit(X_train, y_train)
                
            pred = clf_grid.predict(X_test)
            
            score = precision_recall_fscore_support(y_test, pred, average='weighted')
            results.append((score, ngram, stopword))
            #store the best classifier, its score, stopword setting and ngram values.
            if (score[2]> best_score[2]):
                best_score = score
                best_clf = clf_grid
                stopw = stopword
                conf_matrix = confusion_matrix(y_test, pred)
                ng_val = ngram
                best_y_test = y_test
                final_count_vect = count_vect 
    
    #return the optimal classifiers and its attributes
    return best_clf, [best_score, stopw, ng_val, conf_matrix, best_y_test, pred], results, final_count_vect, lb


def run_naive_bayes(corpus, y_vec, featureset, parameters=None, stopw_list=[False], clf=None, pass_=True, count_vect=None, lb=None):
    """Run naive bayes.
    
    """
    #alpha is the laplace bias
    print('Running Multinomial Naive Bayes. Testing for optimal paramater for alpha range of 0.1 to 2')
    print('Testing: ' + featureset)
    if not clf:
        mlb = MultinomialNB()
    else:
        mlb = clf
    clf2, attributes, mlb_results, final_count_vect, main_lb = find_optimal_params(mlb, 
                                                                                   corpus, 
                                                                                   y_vec, 
                                                                                   parameters, 
                                                                                   stopw_list, 
                                                                                   pass_=pass_, 
                                                                                   final_count_vect=count_vect, 
                                                                                   lb=lb)
    if parameters:
        print('Best variant of the estimator is: ' + str(clf.best_estimator_))
        print('With ngrams setting set to: ' + str(attributes[2]))
        if (attributes[1]):
            print('With removal of stopwords')
        else:
            print('With no removal of stopwords')
        print('F1-Score: '+str(attributes[0][2]) +'\n')
    
    return clf2, attributes, mlb_results, final_count_vect, main_lb


def make_y_list(sub_files, main_corp):
    """Makes a Y vector for training.
    
    Args:
        sub_files: A, Y, B.
        main_corp: Something
        
    Returns:
        sub_files, main_corp
    """
    y_vec = []
    for i in range(len(sub_files)):
        j = len(main_corp[i])
        for e in range(j):
            y_vec.append(sub_files[i])
    return y_vec
