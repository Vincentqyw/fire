

import numpy as np
import pickle
import os
# export to pair files
def pairs_from_similirity(query,db,rank,topK = 10,output_pairs=''):

    db_list,query_list = [],[]
    for image_name in os.listdir(db):
        timestamp = (image_name)
        if timestamp[-4:] == '.png' or timestamp[-4:] == '.jpg':
            db_list.append(os.path.join(db,timestamp))

    for image_name in os.listdir(query):
        timestamp = (image_name)
        if timestamp[-4:] == '.png' or timestamp[-4:] == '.jpg':
            query_list.append(os.path.join(query, timestamp))
    pairs = []
    counter = 0

    for q in query_list:
        image_name = q

        top_similarity_id = rank[counter,0:topK]
        for id in top_similarity_id:
            pair = (image_name, db_list[id])
            pairs.append(pair)
        counter += 1
    with open(output_pairs, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))

if __name__ == "__main__":

    exp_dir     = '/home/slamman/qinyanwen/project_virtual_localization2.0/code/fire/fire/fire_experiments/eval_fire'
    # result_file = exp_dir + '/' + 'query_results_def_night_iphone_x1.pkl'
    # db_path     = '/media/slamman/32AD9115807CA97E/dataset/omni_hik/def_inout/dense/images/db'
    # query_path  = '/media/slamman/32AD9115807CA97E/dataset/omni_hik/localization/def_inout/loc_def_night_slight/iphone12_x1/query'

    result_file = exp_dir + '/' + 'query_results_def_night_insta360.pkl'
    db_path     = '/media/slamman/32AD9115807CA97E/dataset/omni_hik/def_inout/dense/images/db'
    query_path  = '/media/slamman/32AD9115807CA97E/dataset/omni_hik/localization/def_inout/loc_def_night_slight_more/insta360/query'

    output_pairs = os.path.join(exp_dir, 'pairs_def_night_slight_more_insta360_asmk.txt')



    with open(result_file, 'rb') as handle:
        result = pickle.load(handle)


    ranks = result['ranks']

    pairs_from_similirity(query = query_path, db = db_path, rank = ranks, topK=25, output_pairs=output_pairs)
