import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
import torch
import torchvision as tv
from ASL.src.helper_functions.helper_functions import parse_args
from ASL.src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from ASL.src.models import create_model
import argparse
import time
import numpy as np
from scipy.stats import binom
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from utils import *
import seaborn as sns
from core.concentration import romano_wolf_multiplier_bootstrap, bonferroni_HB, uniform_region
from core.uniform_concentration import required_fdp
import pdb

parser = argparse.ArgumentParser(description='ASL MS-COCO predictor')

parser.add_argument('--model_path',type=str,default='./ASL/models_local/MS_COCO_TResNet_xl_640_88.4.pth')
parser.add_argument('--dset_path',type=str,default='../data/')
parser.add_argument('--model_name',type=str,default='tresnet_xl')
parser.add_argument('--input_size',type=int,default=640)
parser.add_argument('--dataset_type',type=str,default='MS-COCO')
parser.add_argument('--th',type=float,default=0.7)

def get_lhat(scores, labels, alpha_plus, num_lam):
    lams = torch.linspace(0,1,num_lam)
    lam = None
    for i in reversed(range(lams.shape[0])):
        lam = lams[i]
        est_labels = (scores > lam).to(float) 
        if est_labels.numel() == 0:
            break
        fdp = (est_labels * labels.to(float)/est_labels.sum()).sum()
        if fdp <= alpha_plus:
            break
    return lam

def get_lhat_from_table(calib_loss_table, lambdas_table, alpha_plus):
    calib_loss_table = calib_loss_table[:,::-1]
    fdrs = calib_loss_table.mean(axis=0)

    for i in reversed(range(1, len(lambdas_table))):
        fdr = fdrs[i]
        if fdr <= alpha_plus:
            return lambdas_table[(-i+1)] #TODO: FIX TODO: FIX TODO: FIX i+2 ; one of the +1 comes from the overshoot of Rhat + t, and the other from 0-indexing. 

    return lambdas_table[0]

def get_example_fdr_and_size_tables(scores, labels, lambdas_example_table):
    lam_len = len(lambdas_example_table)
    lam_low = min(lambdas_example_table)
    lam_high = max(lambdas_example_table)
    fname_loss = f'../.cache/{lam_low}_{lam_high}_{lam_len}_example_fdr_table.npy'
    fname_sizes = f'../.cache/{lam_low}_{lam_high}_{lam_len}_example_size_table.npy'
    try:
        loss_table = np.load(fname_loss)
        sizes_table = np.load(fname_sizes)
    except:
        loss_table = np.zeros((scores.shape[0], lam_len))
        sizes_table = np.zeros((scores.shape[0], lam_len))
        print("caching loss and size tables")
        for j in tqdm(range(lam_len)):
            est_labels = scores > lambdas_example_table[j]
            loss, sizes = get_metrics_precomputed(est_labels, labels)
            loss_table[:,j] = loss 
            sizes_table[:,j] = sizes

        np.save(fname_loss, loss_table)
        np.save(fname_sizes, sizes_table)

    return loss_table, sizes_table

def trial_precomputed(rejection_region_function, rejection_region_name, example_loss_table, example_size_table, lambdas_example_table, alpha, delta, num_lam, num_calib, m):
    rng_state = np.random.get_state()
    np.random.shuffle(example_loss_table)
    np.random.set_state(rng_state)
    np.random.shuffle(example_size_table)

    calib_losses, val_losses = (example_loss_table[0:num_calib], example_loss_table[num_calib:])
    calib_sizes, val_sizes = (example_size_table[0:num_calib], example_size_table[num_calib:])

    if rejection_region_name == "Bardenet (Uniform)":
        R = rejection_region_function(calib_losses,lambdas_example_table,alpha,delta,m)
    else:
        R = rejection_region_function(calib_losses,lambdas_example_table,alpha,delta)

    # Pick the largest set
    if len(R) == 0:
        lhat = -1
    else:
        lhat = lambdas_example_table[min(R)]

    fdrs = val_losses[:,np.argmax(lambdas_example_table == lhat)]
    sizes = val_sizes[:,np.argmax(lambdas_example_table == lhat)]

    return fdrs.mean(), torch.tensor(sizes), float(lhat)

def plot_histograms(df_list,alpha,delta):
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,3))

    sizes = torch.cat(df_list[0]['sizes'].tolist(),dim=0).numpy()
    d = np.diff(np.unique(sizes)).min()
    lofb = sizes.min() - float(d)/2
    rolb = sizes.max() + float(d)/2

    for i in range(len(df_list)):
        df = df_list[i]
        axs[0].hist(np.array(df['FDR'].tolist()), None, alpha=0.7, density=True)

        # Sizes will be 10 times as big as recall, since we pool it over runs.
        sizes = torch.cat(df['sizes'].tolist(),dim=0).numpy()
        axs[1].hist(sizes, np.arange(lofb,rolb+d, d), alpha=0.7, density=True, label=df['region name'][0])
    
    axs[0].set_xlabel('FDR')
    axs[0].locator_params(axis='x', nbins=4)
    axs[0].set_ylabel('density')
    axs[0].axvline(x=alpha,c='#999999',linestyle='--',alpha=0.7)
    axs[1].set_xlabel('size')
    sns.despine(ax=axs[0],top=True,right=True)
    sns.despine(ax=axs[1],top=True,right=True)
    axs[1].legend()
    plt.tight_layout()
    plt.savefig('../' + (f'outputs/histograms/{alpha}_{delta}_coco_histograms').replace('.','_') + '.pdf')


def experiment(rejection_region_functions,rejection_region_names,alpha,delta,num_lam,num_calib,lambdas_example_table,num_trials,coco_val_2017_directory,coco_instances_val_2017_json):
    df_list = []

    for idx in range(len(rejection_region_functions)):
        rejection_region_function = rejection_region_functions[idx]
        rejection_region_name = rejection_region_names[idx]
        fname = f'../.cache/{alpha}_{delta}_{num_calib}_{num_trials}_{rejection_region_name}_dataframe.pkl'

        df = pd.DataFrame(columns = ["$\\hat{\\lambda}$","FDR","sizes","alpha","delta","region name"])
        try:
            df = pd.read_pickle(fname)
        except FileNotFoundError:
            dataset = tv.datasets.CocoDetection(coco_val_2017_directory,coco_instances_val_2017_json,transform=tv.transforms.Compose([tv.transforms.Resize((args.input_size, args.input_size)),tv.transforms.ToTensor()]))
            print('Dataset loaded')
            
            #model
            state = torch.load('./ASL/models_local/MS_COCO_TResNet_xl_640_88.4.pth', map_location='cpu')
            classes_list = np.array(list(state['idx_to_class'].values()))
            args.num_classes = state['num_classes']
            model = create_model(args).cuda()
            model.load_state_dict(state['model'], strict=True)
            model.eval()
            print('Model Loaded')
            corr = get_correspondence(classes_list,dataset.coco.cats)

            # get dataset
            dataset_fname = '../.cache/coco_val.pkl'
            if os.path.exists(dataset_fname):
                dataset_precomputed = pkl.load(open(dataset_fname,'rb'))
                print(f"Precomputed dataset loaded. Size: {len(dataset_precomputed)}")
            else:
                dataset_precomputed = get_scores_targets(model, torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True), corr)
                pkl.dump(dataset_precomputed,open(dataset_fname,'wb'),protocol=pkl.HIGHEST_PROTOCOL)

            scores, labels = dataset_precomputed.tensors
            example_fdr_table, example_size_table = get_example_fdr_and_size_tables(scores, labels, lambdas_example_table)
            print(f'Total samples: {scores.shape[0]}')
            m = scores.shape[1]
            
            local_df_list = []
            for i in tqdm(range(num_trials)):
                fdr, sizes, lhat = trial_precomputed(rejection_region_function, rejection_region_name, example_fdr_table, example_size_table, lambdas_example_table, alpha, delta, num_lam, num_calib, m)
                if lhat < 0:
                    continue
                dict_local = {"$\\hat{\\lambda}$": lhat,
                                "FDR": fdr,
                                "sizes": [sizes],
                                "alpha": alpha,
                                "delta": delta,
                                "region name": rejection_region_name
                             }
                df_local = pd.DataFrame(dict_local)
                local_df_list = local_df_list + [df_local]
            if len(local_df_list) == 0:
                continue
            df = pd.concat(local_df_list, axis=0, ignore_index=True)
            df.to_pickle(fname)
        df_list = df_list + [df]

    plot_histograms(df_list,alpha,delta)


if __name__ == "__main__":
    with torch.no_grad():
        sns.set(palette='pastel',font='serif')
        sns.set_style('white')
        fix_randomness(seed=0)
        args = parse_args(parser)
        coco_val_2017_directory = '../data/val2017'
        coco_instances_val_2017_json = '../data/annotations_trainval2017/instances_val2017.json'

        alphas = [0.05,0.1,0.2,0.5]
        deltas = [0.1,0.1,0.1,0.1]
        params = list(zip(alphas,deltas))
        num_lam = 1500 
        num_calib = 4000 
        num_trials = 5 
        lambdas_example_table = np.linspace(0,1,num_lam)

        rejection_region_functions = (romano_wolf_multiplier_bootstrap, bonferroni_HB, uniform_region)
        rejection_region_names = ('RWMB', 'HBBonferroni', 'Bardenet (Uniform)')
        
        for alpha, delta in params:
            print(f"\n\n\n ============           NEW EXPERIMENT alpha={alpha} delta={delta}           ============ \n\n\n") 
            experiment(rejection_region_functions,rejection_region_names,alpha,delta,num_lam,num_calib,lambdas_example_table,num_trials,coco_val_2017_directory,coco_instances_val_2017_json)
