def calculate_intersection(intersection_list):
    intersection_keys = set()
    for ij in intersection_list:
        if len(intersection_keys) == 0:
            intersection_keys = set(ij)
        else:
            intersection_keys = intersection_keys.union(set(ij))
    return intersection_keys
    

def build_intersection_matrix_of_subreddits(sub_count_list, top_sub_n=2000):
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
