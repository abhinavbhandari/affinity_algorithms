def calculate_intersection(intersection_list):
    """Calculates intersection between two different subreddits user activities."""
    intersection_keys = set()
    for ij in intersection_list:
        if len(intersection_keys) == 0:
            intersection_keys = set(ij)
        else:
            intersection_keys = intersection_keys.union(set(ij))
    return intersection_keys
    

def build_intersection_matrix_of_subreddits(sub_count_list, top_sub_n=2000):
    """Builds intersection matrix of common subreddits visited between different users.
    
    Users from various subreddits may visit similar subreddits. This is an interesting
    finding because it reveals how much overlap there is likely to be despite of politicized
    differences between various groups. As such, this network of most visitations should reveal
    how users across different communities on reddit behave.
    
    Args:
        sub_count_list (list): List of subreddits with visitation counts for each group.
        top_sub_n (int): Number of top subreddits to compare and analyse
        
    Returns:
        int_list (list): A matrix of list.
    """
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
