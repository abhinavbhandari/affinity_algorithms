def transform_sub_count_to_auth_to_dic(total_original_sub_count):
    """
    Purpose:
        Takes in a list of subreddit to author counts, transforms it into
        author to dictionary counts. 
    """
    auth_to_sub = {}
    for sub_index, auth_count_dic in total_original_sub_count.items():
        for auth_index, auth_count in auth_count_dic.items():
            if auth_index not in auth_to_sub:
                auth_to_sub[auth_index] = {}
            auth_to_sub[auth_index][sub_index] = auth_count
    return auth_to_sub


def get_preference_subs(auth_to_sub_by_list, interval=2):
    """
    Purpose:
    Get Loyal authors for each subreddit, and the number of loyal authors a subreddit has.
    
    Parameters:
    auth_to_sub_by_list: a list of auth_to_sub dics for each time interval.
    
    """
    
    preference_count = {}
    #
    for i, auth_dics in enumerate(auth_to_sub_by_list):
        for j, auth in enumerate(auth_dics):
            auth_posts = auth_dics[auth]
            sub_index, preference_bool = get_most_commented_sub(auth_posts)
            if auth in preference_count:
                if sub_index in preference_count[auth]:
                    preference_count[auth][sub_index].append(preference_bool)
                else:
                    preference_count[auth][sub_index] = [preference_bool]
            else:
                preference_count[auth] = {}
                preference_count[auth][sub_index] = [preference_bool]
    
    return preference_count


def get_loyal_subs(preference_count):
    """
    Purpose:
    Extract loyal subs from preferred subs by a user. 
    """
    
    loyalty_count = {}
    loyal_sub_to_auths = {}
    
    for auth, sub_to_pref in preference_count.items():
        for sub_index, pref_lists in sub_to_pref.items():
            loyal = 0
            for pref in pref_lists:
                if pref:
                    loyal += 1
            if loyal > 1:
                if sub_index in loyalty_count:
                    loyalty_count[sub_index] += 1
                else:
                    loyalty_count[sub_index] = 1
                if sub_index in loyal_sub_to_auths:
                    loyal_sub_to_auths[sub_index].append(auth)
                else:
                    loyal_sub_to_auths[sub_index] = [auth]
    
    return loyalty_count, loyal_sub_to_auths


def get_metric_percentage(subset_dic, total_dic):
    """
    Purpose:
    Used for calculating metrics such as percentage of dedicated users or percentage of loyal users
    
    Parameters:
    subset_dic: The dictionary that is going to be used to create numbers
    total_dic: should contain the total number of users (Preferably send sub: number of users)
    """
    
    metric_dictionary = {}
    for sub_index, proportion_of_sub_users in subset_dic.items():
        
        # Remember that right now I am using len, but this should not be the case.
        if sub_index not in total_dic:
            continue
        total_num_of_users = len(total_dic[sub_index])
        percentage = proportion_of_sub_users/total_num_of_users
        metric_dictionary[sub_index] = percentage
    return metric_dictionary


def get_most_commented_sub(auth_posts):
    """
    Purpose:
    See if a user prefers a given sub in their post histories. Also return the sub_index of that subreddit. 
    """
    total_count = 0
    max_count = 0
    max_sub_index = -1
    for sub_index, post_count in auth_posts.items():
        total_count += post_count
        if max_count < post_count:
            max_sub_index = sub_index
            max_count = post_count
    preference_val = max_count/total_count
    preference = False
    if preference_val >= 0.5:
        preference = True
    return max_sub_index, preference


def get_auth_to_subs(sbcs):
    """
    Purpose:
        X, Y, Z: 
    """
    auth_to_subs = []
    for sbc in sbcs:
        auth_to_subs.append(transform_sub_count_to_auth_to_dic(sbc))
    return auth_to_subs


def get_subreddit_to_loyalty(intersect_subs_author_total, sbcs):
    """
    Purpose:
        The main engine function that extracts computes the percentage such that 
       
    """
    auth_to_subs = get_auth_to_subs(sbcs)
    preference_count_subs = get_preference_subs(auth_to_subs)
    loyalty_count, loyalty_sub_to_auths = get_loyal_subs(preference_count_subs)
    loyalty_percent = get_metric_percentage(loyalty_count, intersect_subs_author_total)
    return loyalty_percent