import os
import pickle
import pandas as pd


class UserInfo(object):
    """Encapsulates user information such as comments and subreddits participated in.
    
    User Info object is designed to contain members and functions that extract information
    form the history of a user's posts. This includes information such as how diverse a
    particular user's interests are in terms of the subreddits they participate in, or how
    dense their network of comments are.
    """
    def __init__(self, name, subreddits=None, posts=None, time=None, main_sub=None):
        """Initializes the user object.
        
        Args:
            name (str): Name of the user
            subreddits (list): List of subreddits user participated on
            posts (list): List of comments from the user
            time (list): Date of comments
            main_sub (str): Target sub on which the user was found.
        """
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
        """Public method to add subreddit.
        """
        self.subreddits.append(sub)
        
    def add_info(self, sub, post, d_time):
        """Public method to add information.
        
        Information is a collection of subreddit, comment, and date.
        
        Args:
            sub: Subreddit in str
            post: Comment in str
            date: Datetime object for the comment date
            
        Returns:
            None
        """
        self.subreddits.append(sub)
        self.posts.append(post)
        self.time.append(d_time)
        
        
    def graph_info():
        """Calculates the count of participation for each subreddit.
        """
        self.sub_dic = {}
        for sub in self.sub:
            if sub in sub_dic:
                sub_dic[sub] += 1
            else:
                sub_dic[sub] = 1
    
    def graph_spread():
        """Calculates the number of subreddits participated in.
        """
        if not self.sub_dic:
            self.graph_info()
        
        return len(self.sub_dic)
    
    def make_gephi_node(filtered=None):
        """Computes a list for creation of gephi node.
        
        Gephi is a graph generating software. Its nodes require the creation
        of weighted edges that are ingoing and outgoing. This function handles 
        that process of inward and outward edges.
        
        Args:
            filtered: Default is None. Provides a specific list of subreddits
                that should be counted for inward, outward edges.
            
        Returns:
            None
        """
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


def get_subreddit_posts_from_sub_dic(sub_dic, main_sub=None, dedicated=False):
    """Get subreddit posts for target subreddit.
    
    Subreddit posts from a target subreddit can be used to understand the level
    of language evolution that takes place in the target subreddit over time.
    The way this function is structured is that it can extract posts that are separated
    between comments made by dedicated users and non-dedicated users.
    
    Args:
        sub_dic (dic): A dictionary that maps subreddits -> list of comments by users
        main_sub (str): target subreddit
        dedicated (boolean): A boolean value which signals whether to extract dedicated
            posts or not. 0 - No, 1 - Yes, 2 - Both
            
    Returns:
        list of posts
    """
    if not main_sub:
        raise Exception('Error: provide main_sub')
    info_list = sub_dic[main_sub]
    sub_posts = []
    dedicated_posts = []
    sub_dates = []
    for info in info_list:
        if dedicated == 2:
            sub_posts.append(info[0])
            if len(info[0]) > 4:
                dedicated_posts.append(info[0])
        if dedicated:
            if len(info[0]) > 4:
                sub_posts.append(info[0])
        else:
            sub_posts.append(info[0])
    if dedicated == 2:
        return sub_posts, dedicated_posts
    else:
        return sub_posts


def categorize_user_by_sub_dic(users_list,
                               main_sub=None,
                               dedicated_thresh=5):
    """Converts user objects into a dictionary of User Information.
    
    Extracts information from each UserInfo object and inputs it into
    a dictionary. Filters input by dedicated users.
    
    Args:
        users_list (list): List of users
        main_sub (str): Target subreddit.
        dedicated_thresh (int): Default is 5.
        
    Returns:
        sub_dic: A dictionary that maps subreddit -> list of posts.
    """
    sub_dic = {}
    for username, userobj in users_list.items():
        if main_sub:
            count = 0
            for sub_num in info.subreddits:
                if sub_num == main_sub:
                    count+=1
            if count < dedicated_thesh:
                continue
        
        for i in range(len(info.subreddits)):
            subname = info.subreddits[i]
            if subname in sub_dic:
                sub_dic[subname].append([userobj.posts[i], userobj.time[i]])
            else:
                sub_dic[subname] = [[userobj.posts[i], userobj.time[i]]]
    return sub_dic


def get_dedicated_users(user_dic, main_sub, dedicated=5):
    """Extracts dedicated users from a UserInfo dictionnary.
    
    Args:
        user_dic (dic): Maps usernames to UserInfo objects
        main_sub (str): Target subreddit
        dedicated (int): Default is 5.
        
    Returns:
        dedicated_users (dic): Username maps to UserInfo
    """
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
    """Converts a UserInfo Object to independent outputs.
    
    Unzips userinformation from a list of coupled lists, that contain
    (subreddit, comment, date).
    
    Args:
        user_info (list): list of userinfo, (subreddit, comment, date).
        
    Returns:
        subs (list): List of subreddits
        posts (list): List of posts
        d_time (list): List of Date
    """
    subs = []
    posts = []
    d_time = []
    for info in user_info:
        subs.append(info[0])
        posts.append(info[1])
        d_time.append(info[2])
    return subs, posts, d_time


def user_info_factory(dic, user_dic=None, main_sub=None, dedicated_set=None):
    """A user info factory, that creates user info based on specifications.
    
    There is raw JSON that contains user metadata, however we encapsulate that
    data into UserInfo to be able to run specific functions on the user meta data.
    
    
    Args:
        dic (dic): Subreddit dictionary with username -> User JSON.
        user_dic (dic): User dic that maps username -> UserInfo
        main_sub (str): Target Subreddit
        dedicated_set (set): Default is None. Empty.
        
    Returns:
        user_dic (dic): User dic that maps . username -> UserInfo
    """
    if not dedicated_set:
        if type(user_dic) == type(None):
            user_dic = {}
        for user, info in dic.items():
            if dedicated_set:
                if user not in dedicated_set:
                    continue
            subs, posts, d_time = convert_user_to_lists(info)
            if user in user_dic:
                user_dic[user].extend_info(subs, posts, d_time, main_sub=main_sub)
            else:
                user_dic[user] = UserInfo(name=user,
                                          subreddits=subs,
                                          posts=posts,
                                          time=d_time, main_sub=main_sub)
    return user_dic