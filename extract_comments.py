import sys
import gzip
import json
from collections import defaultdict
import os

def extract_comments(input_file, output_dir, subreddit_list):

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

    input_file = sys.argv[1]
    output_dir =  "/home/ndg/projects/shared_datasets/semantic-shift-by-subreddit/subreddits_oct2014_june2015/"
#    subreddit_list_file = "all_combined.txt"
    subreddit_list_file = "all_subreddits_above_10000_comments.txt"
    subreddit_list = list(set([line.strip() for line in open(subreddit_list_file)]))

    extract_comments(input_file, output_dir, subreddit_list)


#ls -d -1 /home/ndg/arc/reddit/2014/RC_2014-{10..12}*.gz | parallel -j20 --pipe parallel -j100 --no-notice python extract_comments.py
#ls -d -1 /home/ndg/arc/reddit/2015/RC_2015-0{1..6}*.gz | parallel -j20 --pipe parallel -j100 --no-notice python extract_comments.py