import pickle

def pickle_load(path):
    """Loads a pickle for a path.
    
    Args:
        path (str): path
        
    Returns:
        some_data_type 
    """
    file_data = None
    with open(path, 'rb') as f:
        file_data = pickle.load(f)
    f.close()
    return file_data


def pickle_dump(sub, ext, path, data):
    """Dumps a pickle to directed path.
    
    Args:
        sub (str): subreddit
        ext (str): file extension
        path (str): absolute path of the place where the file should be stored.
        data: payload to be stored.
    """
    fname = sub + ext
    with open(os.path.join(path, fname), 'wb') as f:
        pickle.dump(data, f)
    f.close()