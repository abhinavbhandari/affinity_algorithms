import sys
import gzip
import json
from collections import defaultdict
import os

def extract_comments(input_file, output_dir, subreddit_list):
    """Extracts comments from a subreddit's JSONS.
    
    Data for extraction is retrieved from paths passed to this file.
    
    Args:
        input_file (str): path of files
        output_dir (str): directory to store extracted comments
        subreddit_list (str): List of subreddits to analyse
        
    Saves:
        Writes file to output_dir with subdirectory for each subreddit. 
    """
    print(input_file)
    comm_d = defaultdict(lambda: [])
    fop = gzip.open(input_file, 'rb')
    for line in fop:
        try:
            comm = json.loads(line)
        except:
            print(input_file, line)
            continue

        subreddit = comm['subreddit']

        if subreddit in subreddit_list:
            comm_d[subreddit].append(comm)

    for subreddit in comm_d:
        sub_output_dir = output_dir + subreddit + "/"
        if not os.path.exists(sub_output_dir):
            os.makedirs(sub_output_dir)
        output_f = sub_output_dir + input_file.split("/")[-1]
        with open(output_f, "wb") as outfile:
            for e in comm_d[subreddit]:
                json.dump(e, outfile)
                outfile.write('\n')


if __name__ == '__main__':
    """Main method is invoked for when running with GNU parallel through cmdline.
    
    Args:
        sys.argv[1]: subreddit name
    """
    input_file = sys.argv[1]
    output_dir =  "/home/ndg/projects/shared_datasets/semantic-shift-by-subreddit/subreddits_oct2014_june2015/"
    subreddit_list_file = "all_subreddits_above_10000_comments.txt"
    subreddit_list = list(set([line.strip() for line in open(subreddit_list_file)]))

    extract_comments(input_file, output_dir, subreddit_list)
