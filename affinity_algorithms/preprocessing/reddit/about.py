#-----------------------------
# Get the about.json for each subreddit
#-----------------------------

import os
import time
import requests
import json

headers = {"User-Agent": "lets get these pages"}

sub_path = '/home/ndg/users/carmst16/slang/all_subreddits_above_10000_comments.txt'
data_dir = '/home/ndg/projects/shared_datasets/slang-subreddit-metadata'

with open(sub_path, 'r') as fin:
    all_subs = fin.readlines()
    all_subs = [x.strip() for x in all_subs]

def get_about(sub_name):
    """Retrieves about information from Reddit about json.
    
    Args:
        sub_name (str): Name of subreddit
        
    Saves:
        about content to the target directory.
    """
    print(sub_name)
    url = "http://www.reddit.com/r/{}/about.json".format(sub_name)
    resp = requests.get(url, headers=headers)
    if not resp.ok:
        # handle request error, return -1?
        return -1
    content = resp.json()
    file_name = sub_name+'.json'
    file_path = os.path.join(data_dir, file_name)
    with open(file_path, 'w') as fp:
        json.dump(content, fp)
        return
    
for sub in all_subs:
    get_about(sub)
