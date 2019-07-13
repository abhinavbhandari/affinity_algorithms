import os
import pickle
import pandas as pd

class RedditPost:
    def __init__(self, sub, post, d_time, main_sub=None):
        self.main_sub = main_sub
        self.sub = sub
        self.post = post
        self.time = d_time

class UserInfo(object):
    def __init__(self, name, subreddits=None, posts=None, time=None, main_sub=None):
        
        if not subreddits:
            subreddits = []
        if not posts:
            posts = []
        if not time:
            time = []
        self.username = name
        self.subreddits = subreddits
        self.posts = posts
        self.time = time
        self.sub_dic = None
        self.count = 0
        if main_sub:
            for sub in self.subreddits:
                if sub == main_sub:
                    self.count += 1
    
    def add_subreddit(self, sub):
        self.subreddits.append(sub)
        
    def add_info(self, sub, post, d_time):
        self.subreddits.append(sub)
        self.posts.append(post)
        self.time.append(d_time)
        
    def extend_info(self, subs, posts, d_time, main_sub=None):
        self.subreddits.extend(subs)
        self.posts.extend(posts)
        self.time.extend(d_time)
        if main_sub:
            for sub in subs:
                if main_sub == sub:
                    self.count += 1
        
    
    def graph_info():
        self.sub_dic = {}
        for sub in self.sub:
            if sub in sub_dic:
                sub_dic[sub] += 1
            else:
                sub_dic[sub] = 1
    
    def graph_spread():
        if not self.sub_dic:
            self.graph_info()
        
        return len(self.sub_dic)
    
    def make_gephi_node(filtered=None):
        gephi_list = []
        for sub in self.sub:
            if not filtered:
                text = main_sub,sub
                gephi_list.append(text)
            else:
                if sub in filtered:
                    text=main_sub,sub
                    gephi_list.append(text)
        return gephi_list


def pickle_load(path):
    """
    Purpose:
    A wrapper that takes in a path to a pickle file, loads it and returns it in an object.
    """
    some_data_type = None
    with open(path, 'rb') as f:
        some_data_type = pickle.load(f)
    f.close()
    return some_data_type

def pickle_dump(sub, ext, path, data):
    fname = sub + ext
    with open(os.path.join(path, fname), 'wb') as f:
        pickle.dump(data, f)
    f.close()

def get_subreddit_posts_from_sub_dic(sub_dic, main_sub=None, dedicated=False):
    """
    Purpose:
    
    """
    if not main_sub:
        raise Exception('Error: provide main_sub')
    info_list = sub_dic[main_sub]
    sub_posts = []
    sub_posts_d = []
    sub_dates = []
    for info in info_list:
        if dedicated == 2:
            sub_posts.append(info[0])
            if len(info[0]) > 4:
                sub_posts_d.append(info[0])
        if dedicated:
            if len(info[0]) > 4:
                sub_posts.append(info[0])
        else:
            sub_posts.append(info[0])
    if dedicated == 2:
        return sub_posts, sub_posts_d
    else:
        return sub_posts

def categorize_user_by_sub_dic(users_list, main_sub=None, df_index=['subreddit_name', 'posts', 'date']):
    """
    Purpose:
    Takes a list of user objects, and returns a dictionary with key as subreddit num, and value as post,date.
    Return type:
    sub_num -> [[post_num, date], ...]
    """
    subs = []
    posts = []
    d_time = []
    sub_dic = {}
    for user, info in users_list.items():
        if main_sub:
            count = 0
            for sub_num in info.subreddits:
                if sub_num == main_sub:
                    count+=1
            if count < 5:
                continue
        
        for i in range(len(info.subreddits)):
            subname = info.subreddits[i]
            if subname in sub_dic:
                sub_dic[subname].append([info.posts[i], info.time[i]])
            else:
                sub_dic[subname] = [[info.posts[i], info.time[i]]]
    return sub_dic
#         subs.extend(info.subreddits)
#         posts.extend(info.posts)
#         d_time.extend(info.time)
    
#     return pd.DataFrame(columns=[subs, posts, d_time], index=['subreddit_name', 'posts', 'date']).T

def get_dedicated_users(user_dic, main_sub, dedicated=5):
    dedicated_users = {}
    for username, userobject in user_dic.items():
        count = 0
        for sub in userobject.subreddits:
            if sub == main_sub:
                count += 1
        if count < dedicated:
            continue
        else:
            dedicated_users[username] = userobject
    return dedicated_users
            
def convert_user_to_lists(user_info):
    subs = []
    posts = []
    d_time = []
    for info in user_info:
        subs.append(info[0])
        posts.append(info[1])
        d_time.append(info[2])
    return subs, posts, d_time

# def from_users_to_subreddit_counts(users_dic):
#     sub_dic = []
#     for user in users_list:
#         subs = user.subreddits
#         for sub in subs:
#             if sub in sub_dic:

def create_user_info(dic, user_dic=None, main_sub=None, dedicated_set=None):
    """
    Purpose:
    
    Parameters:
    dic: only takes user dictionary. 
    """
    if not dedicated_set:
        if type(user_dic) == type(None):
            user_dic = {}
        for user, info in dic.items():
            subs, posts, d_time = convert_user_to_lists(info)
            if user in user_dic:
                user_dic[user].extend_info(subs, posts, d_time, main_sub=main_sub)
            else:
                user_dic[user] = UserInfo(name=user,
                                          subreddits=subs,
                                          posts=posts,
                                          time=d_time, main_sub=main_sub)
    if dedicated_set:
        if type(user_dic) == type(None):
            user_dic = {}
        for user, info in dic.items():
            if user in dedicated_set:
                subs, posts, d_time = convert_user_to_lists(info)
                if user in user_dic:
                    user_dic[user].extend_info(subs, posts, d_time)
                else:
                    user_dic[user] = UserInfo(name=user,
                                              subreddits=subs,
                                              posts=posts,
                                              time=d_time)

def extract_files(file_paths, mult=False):
    """
    Purpose: 
    Takes in list of files, then load them into a list and returns them. 
    
    """
    if mult:
        some_data_type = []
        for fn in file_paths:
            some_data_type.append(pickle_load(fn))
    else:
        some_data_type = pickle_load(file_paths)
     
    return some_data_type
        
def get_file_path(path, file_type=None, mult=False):
    """
    Purpose:
    It will return the absolute file paths that are contained inside the path provided. It also returns paths
    for files that contain the file_type name condition. 
    If no file_type is specified, it will return everything that is not a directory. 
    It takes in a list as a path type.
    """
    r_files = []
    for file_path in os.listdir(path):
        new_path = '/'.join([path, file_path])
        if os.path.isdir(new_path):
            r_files.extend(get_file_path(new_path, file_type))
        else:
            if file_type:
                f_bool = False
                for f_t in file_type:
                    if f_t in file_path:
                        f_bool = True
                    else:
                        f_bool = False
                        break
                if f_bool:
                    r_files.append(new_path)
            else:
                r_files.append(new_path)
    return r_files

def make_file_to_dic(path, ftype=None, mult=False):
    file_names = get_file_path(path, ftype)
    extracted = extract_files(file_names, mult=mult)
    return extracted

def combine_users(list_of_user_dics):
    new_user_dic = {}
    for user_dic in list_of_user_dics:
        for user in user_dic:
            if user in new_user_dic:
                new_user_dic[user] = user_dic[user]
            else:
                new_user_dic[user].subs.extend(user_dic[user].subs)
                

def combine_term_counts(extracted):
    """
    Purpose:
    Combines the data from 'subreddit_counts' and returns it in a new dictionary. 
    """
    new_dic = {}
    for dic in extracted:
        for term in dic:
            if term in new_dic:
                new_dic[term.lower()] += dic[term]
            else:
                new_dic[term.lower()] = dic[term]
    return new_dic

def filter_word_counts(word_count): 
    """
    Purpose:
    Receives word_counts dictionary, and returns a filtered one with unnecessary words removed. 
    """
    new_dic = {}
    

def get_files_by_date(subname, ftype=None, years='', months='', days='', 
                        by_year=False, by_month=False, by_day=False):
    """
    Purpose:
    
    Parameters:
    
    """
    path = '/home/ndg/users/abhand1/subreddit_data/all'
    new_word_to_num = []
    paths = []
    extracted = []
    if by_year:
        for year in years:
            year_path = '/'.join([path, year])
            if by_month:
                for month in months:
                    month_path = '/'.join([year_path, month])
                    extracted.extend(make_file_to_dic(month_path, ftype=ftype, mult=True))
            else:
                extracted.extend(make_file_to_dic(month_path, ftype=ftype, mult=True))
    return extracted
#     if len(paths) == 0:
#         print("Error, no year/month/day value submitted")
#         return 
#     return make_file_to_dic(paths, ftype=ftype, mult=True)

def get_sub_id(info_list):
    """
    Purpose:
    Receives info of a user from user_to_sub, returns the subreddit number. 
    """
    subs = []
    for info in info_list:
        subs.append(info[0])
    return subs

def get_sub_id_from_users(dic, user_to_subs=None):
    """
    Purpose: 
    Pass a dictionary of user_to_subreddits, and return a dictionary of username mapped to UserInfo with subreddit ids
    of the user stored (initialized). 
    """
    if not user_to_subs:
        user_to_subs = {}
    for user in dic:
        if not user in user_to_subs:
            user_to_subs[user] = UserInfo(user)
        subs = get_sub_num(dic[user])
        for sub in subs:
            user_to_subs[user].add_subreddit(sub)

def list_of_dics_iterator(extracted, user_info=None, func=None, dedicated_set=None, main_sub=None):
    """
    Purpose:
    Combines extracted dictionaries. 
    """
    user_counts = {}
    if type(user_info) == type(None):
        user_info={}
    for dic in extracted:
        if dedicated_set:
            func(dic, user_info, main_sub, dedicated_set)
        else:
            func(dic, user_info, main_sub, dedicated_set)
    return user_info

def calculate_intersection(intersection_list):
    intersection_keys = set()
    for ij in intersection_list:
        if len(intersection_keys) == 0:
            intersection_keys = set(ij)
        else:
            intersection_keys = intersection_keys.union(set(ij))
    return intersection_keys
    

def build_intersection_matrix_of_subreddits(sub_count_list, top_sub_n=2000):
    # list of count of subreddits
    sorted_keys = []
    for sub_counts in sub_count_list:
        sorted_keys.append(set(sorted(sub_counts, key=lambda x: sub_counts[x], reverse=True)[:top_sub_n]))
    
    int_list = []
    for i, s_key in enumerate(sorted_keys):
        temp_key_list = []
        temp_key_list.extend(sorted_keys[:i])
        temp_key_list.extend(sorted_keys[i+1:])
        temp_key_set = calculate_intersection(temp_key_list)
        sub_set = set(s_key)
        int_count = len(sub_set.intersection(temp_key_set))
        int_list.append(int_count)
    return int_list
        
    
    