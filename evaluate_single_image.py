# Copyright (C) 2021-2022 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# vincentqin added this file to inference single image
# Date: 2022.03.02

import sys
import argparse
from pathlib import Path
import yaml 
import ast

import torch
from torchvision import transforms


import numpy as np
import time
import os


from lib.how.how.utils import io_helpers, logging, logging
from lib.how.how.stages.evaluate import  eval_asmk_fire, load_dataset_fire
from lib.cnnimageretrieval.cirtorch.datasets.genericdataset import ImagesFromList


import fire_network


def evaluate_demo_fire_asmk(demo_eval, evaluation, globals):
    globals["device"] = torch.device("cpu")

    logger = globals["logger"]
    logger.info("Starting global evaluation")

    if demo_eval['gpu_id'] is not None:
        globals["device"] = torch.device(("cuda:%s" % demo_eval['gpu_id']))

    # Handle net_path when directory
    net_path = Path(demo_eval['exp_folder']) / demo_eval['net_path']
    if net_path.is_dir() and (net_path / "epochs/model_best.pth").exists():
        net_path = net_path / "epochs/model_best.pth"

    # Load net
    state = torch.load(net_path, map_location='cpu')
    state['net_params']['pretrained'] = None # no need for imagenet pretrained model
    net  = fire_network.init_network(**state['net_params']).to(globals['device'])
    net.load_state_dict(state['state_dict'])

    globals["transform"] = transforms.Compose([transforms.ToTensor(), \
                transforms.Normalize(**dict(zip(["mean", "std"], net.runtime['mean_std'])))])

    print("==> init FIRe net succeed!")

    # TODO: codebook trainning
    eval_asmk_fire(net, evaluation['inference'], globals, **evaluation['local_descriptor'])

    print("==> Inference done!")


def evaluate_demo_global(demo_eval, evaluation, globals, save_feats=True,load_feats=False):
    globals["device"] = torch.device("cpu")

    logger = globals["logger"]
    logger.info("Starting global evaluation")

    if demo_eval['gpu_id'] is not None:
        globals["device"] = torch.device(("cuda:%s" % demo_eval['gpu_id']))

    # Handle net_path when directory
    net_path = Path(demo_eval['exp_folder']) / demo_eval['net_path']
    if net_path.is_dir() and (net_path / "epochs/model_best.pth").exists():
        net_path = net_path / "epochs/model_best.pth"

    # Load net
    state = torch.load(net_path, map_location='cpu')
    state['net_params']['pretrained'] = None # no need for imagenet pretrained model
    net = fire_network.init_network(**state['net_params']).to(globals['device'])
    net.load_state_dict(state['state_dict'])
    globals["transform"] = transforms.Compose([transforms.ToTensor(), \
                transforms.Normalize(**dict(zip(["mean", "std"], net.runtime['mean_std'])))])

    print("==> init FIRe net succeed!")
    print("==> root path is: " , globals['root_path'])

    # set db and query list to inference
    dataset = 'global descriptors'
    images, qimages, bbxs, gnd = load_dataset_fire(db_path = globals['db_path'], query_path = globals['query_path'])

    logger.info(f"Evaluating {dataset}")

    inference = evaluation['inference']

    # load data or extract global
    if load_feats:
        vecs = np.load(f'{globals["exp_path"]}/db.npy')
        qvecs = np.load(f'{globals["exp_path"]}/query.npy')
    else:
        with logging.LoggingStopwatch("extracting database images", logger.info, logger.debug):
            dset = ImagesFromList(root='', images=images, imsize=inference['image_size'], bbxs=None, transform=globals['transform'])
            vecs = fire_network.extract_vectors(net, dset, globals["device"], scales=inference['scales'])
        with logging.LoggingStopwatch("extracting query images", logger.info, logger.debug):
            qdset = ImagesFromList(root='', images=qimages, imsize=inference['image_size'], bbxs=bbxs, transform=globals['transform'])
            qvecs = fire_network.extract_vectors(net, qdset, globals["device"], scales=inference['scales'])
        vecs, qvecs = vecs.numpy(), qvecs.numpy()

    ranks = np.argsort(-np.dot(vecs, qvecs.T), axis=0)
    ranks = ranks.transpose()

    pairs_from_similirity(qimages, 
                          images, 
                          ranks, 
                          topK=25, 
                          output_pairs=f'{globals["exp_path"]}/pairs_query.txt')

    # export image pairs to file
    if save_feats:

        print("db.shape = ", vecs.shape)
        print("query.shape = ",qvecs.shape)
        print("ranks.shape = ",ranks.shape)

        np.save(f'{globals["exp_path"]}/db.npy',vecs)
        np.save(f'{globals["exp_path"]}/query.npy',qvecs)




# export to pair files
def pairs_from_similirity(query,db,rank,topK = 10,output_pairs=''):
    pairs = []
    counter = 0
    for q in query:
        image_name = q

        top_similarity_id = rank[counter,0:topK]
        for id in top_similarity_id:
            pair = (image_name, db[id])
            pairs.append(pair)
        counter += 1
    with open(output_pairs, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))



def main(args):
    """Argument parsing and parameter preparation for the demo"""
    # Arguments
    parser = argparse.ArgumentParser(description="FIRe evaluation.")
    parser.add_argument('parameters', type=str, help="Relative path to a yaml file that contains parameters.")
    parser.add_argument("--experiment", "-e", metavar="NAME", dest="experiment")
    parser.add_argument("--model-load", "-ml", metavar="PATH", dest="demo_eval.net_path")
    parser.add_argument("--features-num", metavar="NUM",
                        dest="evaluation.inference.features_num", type=int)
    parser.add_argument("--scales", metavar="SCALES", dest="evaluation.inference.scales",
                        type=ast.literal_eval)
    args = parser.parse_args(args)

    # Load yaml params
    package_root    = Path(__file__).resolve().parent
    parameters_path = args.parameters
    params          = io_helpers.load_params(parameters_path)
    # Overlay with command-line arguments
    for arg, val in vars(args).items():
        if arg not in {"command", "parameters"} and val is not None:
            io_helpers.dict_deep_set(params, arg.split("."), val)
            
    # Resolve experiment name
    exp_name = params.pop("experiment")
    if not exp_name:
        exp_name = Path(parameters_path).name[:-len(".yml")]

    # Resolve data folders
    globals = {}
    globals["root_path"] = (package_root / params['demo_eval']['data_folder'])
    globals["root_path"].mkdir(parents=True, exist_ok=True)
    # _overwrite_cirtorch_path(str(globals['root_path']))
    globals["exp_path"] = (package_root / params['demo_eval']['exp_folder']) / exp_name
    globals["exp_path"].mkdir(parents=True, exist_ok=True)
    # Setup logging
    globals["logger"] = logging.init_logger(globals["exp_path"] / f"eval.log")

    # Run demo
    io_helpers.save_params(globals["exp_path"] / f"eval_params.yml", params)
    params['evaluation']['global_descriptor'] = dict(datasets=[])
    # download.download_for_eval(params['evaluation'], params['demo_eval'], DATASET_URL, globals)

    db_path      = '/home/realcat/Datasets/Aachen-Day-Night/images/images_upright/db'
    query_path   = '/home/realcat/Datasets/Aachen-Day-Night/images/images_upright/query/night/nexus5x'
    
    # to train a large asmk dict
    asmk_db_path = '/home/realcat/Datasets/Aachen-Day-Night/images/images_upright/db' 

    globals['asmk_db_path'] =  asmk_db_path
    globals['db_path']      =  db_path
    globals['query_path']   =  query_path
    globals['cache_path']   =  globals["exp_path"]  / "asmk_codebook.bin"

    globals['query_asmk_cache'] = "query_asmk.pkl"
    globals['db_asmk_cache']    = "db_asmk.pkl"


    print(globals['cache_path'])

    # fire asmk test
    # evaluate_demo_fire_asmk(**params, globals=globals)

    # global descriptor
    evaluate_demo_global(**params, globals=globals)



if __name__ == "__main__":
    main(sys.argv[1:])