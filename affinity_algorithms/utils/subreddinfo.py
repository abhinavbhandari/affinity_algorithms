import os
import pickle
import pandas as pd


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
