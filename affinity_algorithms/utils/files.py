import pickle
import os


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
    
    
def extract_files(file_paths, mult=False):
    """Takes in list of files, then load them into a list and returns them. 
    
    Args:
        file_paths (list): List of paths
        mult (bool): Default is False. Runs it on multiple file paths.
        
    Returns:
        file_data (list): Data of files.
    """
    if mult:
        file_data = []
        for fn in file_paths:
            file_data.append(pickle_load(fn))
    else:
        file_data = pickle_load(file_paths)
     
    return file_data


def get_file_path(path, file_type=None, mult=False):
    """Returns absolute file paths inside the path provided.
    
    It will return the absolute file paths that are contained inside the path provided. 
    It also returns paths for files that contain the file_type string condition. 
    If no file_type is specified, it will return everything that is not a directory. 
    It takes in a list as a path type.
    
    Args:
        path (str): parent path from which paths are to be returned
        file_type (list): List of types of str to look for
        mult (bool): Default is false. Run on multiple paths
        
    Returns:
        r_files (list): List of file paths.
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
    """Extracts file paths from a provided path.
    """
    file_names = get_file_path(path, ftype)
    extracted = extract_files(file_names, mult=mult)
    return extracted


def get_files_by_date(ftype=None, years='', months='', days='', 
                        by_year=False, by_month=False, by_day=False):
    """Filter files from a subreddit by date
    
    Args:
        ftype (list): List of types of files to extract. Default is None
        years (list): List of years directory to extract from
        months (list): List of months directory to extract from
        days (list): List of days directory to extract from
        
    Returns:
        extracted (list): All the paths that have been searched for.
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
