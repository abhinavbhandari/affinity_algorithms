import glob
from subreddinfo import pickle_load


class SubredditMetrics(object):
    
    def __init__(self, sbcs, n=10000):
        """Returns a list that contains dedicated, users, comments, and loyalty.
        
        Returns a list that contains total_dedicated, intersect_total_users, 
        intersect_total_comments and intersect_loyalty. Also INTERSECT means exists 
        across different time periods. In other words, an intersecting user is a user 
        that participates on reddit on all the given time periods.
        """
        return_vals = retrieve_user_metrics(sbcs, n)
        self.intersect_dedicated = return_vals[0]
        self.intersect_users = return_vals[1]
        self.intersect_comments = return_vals[2]
        self.intersect_loyalty = return_vals[3]
        
        results_list = self.get_subreddit_rankings_by_metrics(self.intersect_comments, self.intersect_loyalty, self.intersect_dedicated, self.intersect_users)
        
        self.sub_indexes = results_list[0]
        
    
    def get_hundred_subs(val_list, sub_name_list, anchor_index):
        diff = []
        top_100_indexes = []
    #     anchor_index = 36
        below_index = anchor_index - 1
        above_index = anchor_index + 1
        for i in range(100):

            if below_index > 0:
                above_diff = abs(val_list[above_index] - val_list[anchor_index])
                below_diff = abs(val_list[below_index] - val_list[anchor_index])

                if above_diff < below_diff:
                    anchor_index = above_index
                    above_index = anchor_index + 1
                else:
                    anchor_index = below_index
                    below_index = below_index - 1

            elif below_index <= 0:
                anchor_index = above_index
                above_index = above_index + 1
            top_100_indexes.append(anchor_index)
        return top_100_indexes
    
    
    def get_subreddit_rankings_by_metrics(self, intersect_total_comments, 
                                          intersect_loyalty, total_dedicated, 
                                          intersect_total_users):
        loyalty_vals = []
        dedicated_vals = []
        comment_vals = []
        user_vals = []

        storing_sub_indexes = []
        
        subreddits = intersect_total_comments.keys()
        for sub in subreddits:
            if sub in intersect_loyalty and sub in total_dedicated and sub in intersect_total_users:
                loyalty_vals.append(intersect_loyalty[sub])
                dedicated_vals.append(total_dedicated[sub])
                comment_vals.append(intersect_total_comments[sub])
                user_vals.append(intersect_total_users[sub])
                storing_sub_indexes.append(sub)

        return storing_sub_indexes, loyalty_vals, dedicated_vals, comment_vals, user_vals

        

def load_subreddit_counts(load_dir_path='/home/ndg/users/abhand1/subreddit_data/all/global/', load_file_type='subreddit_count', subreddit_counts_exts=['201411', '201412', '201501', '201502', '201503', '201504', '201505', '201506']):
    """
    Purpose:
        Function returns the the subreddit counts of a specified time period. These results are loaded from a file
        that, hass a default structure of subreddit_counts. 
    """
    sbcs = []
    for file_name in glob.glob(load_dir_path + '*' + load_file_type + '*'):
        for ext in subreddit_counts_exts:
            if ext in file_name:
                sbcs.append(pickle_load(file_name))
    
    return sbcs


def save_subreddit_counts():
    """
    Purpose:
        Save subreddit counts.
    """


def subs_in_time_frame(subs_all_months):
    """
    Purpose:
    Extracts subreddits that are in a particular time frame for all months.
    
    """
    intersecting_subs = set(subs_all_months[0].keys())
    for sub_month in subs_all_months[1:]:
        sub_keys = set(sub_month.keys())
        intersecting_subs = intersecting_subs.intersection(sub_keys)
    return intersecting_subs


def subs_more_than_n(total_sub_count, min_n=10000):
    """
    Purpose:
    
    
    """
    new_total_sub_count = {}
    for sub_n, count in total_sub_count.items():
        if count >= min_n:
            new_total_sub_count[sub_n] = count
    return new_total_sub_count


def calculate_total_number_of_comments_in_subs(subs_to_auths):
    """
    Purpose:
    Calculate the total number of comments in a subreddit. Iterates through the total number of comments made
    to a specific subreddit by each author that has commented on said specific subreddit."""
    sub_total = {}
    
    for sub_index, auth_to_count in subs_to_auths.items():
        for auth, count in auth_to_count.items():
            if sub_index in sub_total:
                sub_total[sub_index] += count
            else:
                sub_total[sub_index] = count
    return sub_total


def get_total_posts_for_author_in_subs(original_sub_count, intersect_subs=None):
    """
    Purpose:
    
    
    """
    
    total_count = {}
    for partial_sc in original_sub_count:
        # If we are going to get the total of all posts by an author. 
        if intersect_subs:
            iter_subs = intersect_subs
        else:
            iter_subs = partial_sc
            
        for isub_name in iter_subs:
            
            if isub_name not in total_count:
                total_count[isub_name] = {}
            
            for auth_counts, val in partial_sc[isub_name].items():
                if auth_counts in total_count[isub_name]:
                    total_count[isub_name][auth_counts] += val
                else:
                    total_count[isub_name][auth_counts] = val
    
    return total_count


def get_number_of_dedicated_users(total_freq, n=10):
    """
    Purpose:
    
    """
    total_ave = {}                
    for sub_n, sb in total_freq.items():
        sb_len = len(sb)
        total_val = 0 
        for unames in sb:
            if sb[unames] >= n:
                total_val += 1
        total_ave[sub_n] = total_val/sb_len
    return total_ave


def get_total_number_of_users(subs_to_auths):
    """
    Purpose:
    
    """
    return {sub_index: len(auth_to_dics) for sub_index, auth_to_dics in subs_to_auths.items()}


def retrieve_user_metrics(sbcs, n=10000):
    """
    Function returns user metrics, such as loyalty, dedication, comments and users across different time periods. 
    
    
    """
    intersect_subs = subs_in_time_frame(sbcs)

    # should include the total for intersecting subs. 
    intersect_subs_author_total = get_total_posts_for_author_in_subs(sbcs, intersect_subs)

    intersect_total_comments = calculate_total_number_of_comments_in_subs(intersect_subs_author_total)


    # Fix the following function and clean it. 
    intersect_sub_auth_total_more_than_n = subs_more_than_n(intersect_total_comments, min_n=n)


    intersect_subs_author_total = get_total_posts_for_author_in_subs(sbcs, intersect_sub_auth_total_more_than_n.keys())


    intersect_total_users = get_total_number_of_users(intersect_subs_author_total)


    total_dedicated = get_number_of_dedicated_users(intersect_subs_author_total)
    
    intersect_loyalty = extract_loyal_users.get_subreddit_to_loyalty(intersect_subs_author_total, sbcs)
    
    return total_dedicated, intersect_total_users, intersect_total_comments, intersect_loyalty
