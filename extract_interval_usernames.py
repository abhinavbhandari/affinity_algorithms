# coding=utf-8
import json
import sys
import glob

def extract_username_lists(files, outfile):
    """
    Given a list of files containing Reddit comments for an interval,
    extracts all usernames and writes them to a file.

    :param file_directory:
    :param outfile
    :return:
    """

    authors = []
    for filepath in files:
        with open(filepath, "r") as f:
            for line in f:
                try:
                    comm = json.loads(line)
                except:
                    print(filepath, line)
                    continue

                author = comm["author"]

                authors.append(author)


    with open(outfile, "w") as f:

        f.write(" ".join(authors))


if __name__ == "__main__":

    subreddit_data_dir = sys.arv[1]

    subreddit = subreddit_data_dir.split("/")[-1]
    output_directory = "/home/ndg/projects/shared_datasets/semantic-shift-by-subreddit/subreddits_oct2014_june2015_author/" + subreddit + "/"

    intervals = ["RC_2014-10",
                 "RC_2014-11",
                 "RC_2014-12",
                 "RC_2015-01",
                 "RC_2015-02",
                 "RC_2015-03",
                 "RC_2015-04",
                 "RC_2015-05",
                 "RC_2015-06"]

    for interval in intervals:
        filestring = subreddit_data_dir + "/" + interval + "*"
        files = glob.glob(filestring)
        output_dir = output_directory + interval + "_authors.txt"
