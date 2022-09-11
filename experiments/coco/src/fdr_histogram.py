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
from core.concentration import oracle_HB, romano_wolf_multiplier_bootstrap, romano_wolf_HB, bonferroni_HB, bonferroni_search_HB, multiscale_bonferroni_search_HB, uniform_region
import pdb

parser = argparse.ArgumentParser(description='ASL MS-COCO predictor')

parser.add_argument('--model_path',type=str,default='~/Code/conformal-prediction/data/coco/MS_COCO_TResNet_xl_640_88.4.pth')
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
            return lambdas_table[(-i+1)] 

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

        os.makedirs('../.cache',exist_ok=True)
        np.save(fname_loss, loss_table)
        np.save(fname_sizes, sizes_table)

    return loss_table, sizes_table

def trial_precomputed(rejection_region_function, rejection_region_name, example_loss_table, example_size_table, lambdas_example_table, alpha, delta, num_lam, num_calib, m):
    rng_state = np.random.get_state()
    np.random.shuffle(example_loss_table)
    np.random.set_state(rng_state)
    np.random.shuffle(example_size_table)

    # Big lambda = Big loss
    example_loss_table = example_loss_table[:,::-1]
    example_size_table = example_size_table[:,::-1]

    calib_losses, val_losses = (example_loss_table[0:num_calib], example_loss_table[num_calib:])
    calib_sizes, val_sizes = (example_size_table[0:num_calib], example_size_table[num_calib:])

    if rejection_region_name == "Uniform":
        R = rejection_region_function(calib_losses,lambdas_example_table,alpha,delta,m)
    else:
        R = rejection_region_function(calib_losses,lambdas_example_table,alpha,delta)

    # Pick the largest set
    if len(R) == 0:
        lhat = -1
    else:
        lhat = lambdas_example_table[max(R)]

    fdrs = val_losses[:,np.argmax(lambdas_example_table == lhat)]
    sizes = val_sizes[:,np.argmax(lambdas_example_table == lhat)]

    return fdrs.mean(), torch.tensor(sizes), float(lhat)

def table_function(sizes_array,labels):
    strng = ""
    for i in range(len(labels)):
        strng = strng + f"{labels[i]} & {int(np.median(sizes_array[i]))} & {int(np.quantile(sizes_array[i],0.75))} & {int(np.quantile(sizes_array[i], 0.9))} & {int(np.quantile(sizes_array[i],0.99))} & {int(np.quantile(sizes_array[i],0.999))}"
        strng += "\\\\\n"
    return strng 

def plot_histograms(df_list,alpha,delta):
    sizes = torch.cat(df_list[0]['sizes'].tolist(),dim=0).numpy()
    d = np.diff(np.unique(sizes)).min()
    lofb = sizes.min() - float(d)/2
    rolb = sizes.max() + float(d)/2

    fdr_arrays = []
    sizes_arrays = []
    labels = []
    for i in range(len(df_list)):
        df = df_list[i]
        #fdrs = df['FDR'][df['FDR'] > alpha/2]
        fdrs = df['FDR']
        fdr_arrays = fdr_arrays + [np.array(fdrs.tolist()),]

        # Sizes will be 10 times as big as recall, since we pool it over runs.
        sizes = torch.cat(df['sizes'].tolist(),dim=0).numpy()
        sizes_arrays = sizes_arrays + [sizes,]
        labels = labels + [df['region name'][0],]

    # First plot: violins
    fig = plt.figure(figsize=(6,3))
    ax = fig.gca()
    sns.violinplot(data=fdr_arrays, ax=ax,orient='h',inner=None)
    ax.set_xlabel('FDR', fontsize=14)
    ax.locator_params(axis='x', nbins=4)
    ax.set_yticklabels(labels)
    ax.axvline(x=alpha,c='#999999',linestyle='--',alpha=0.7)
    ax.tick_params(axis='both', labelsize=14)
    sns.despine(ax=ax,top=True,right=True)
    plt.tight_layout()
    os.makedirs('../outputs',exist_ok=True)
    plt.savefig('../' + (f'outputs/violins_tables/{alpha}_{delta}_coco_violins').replace('.','_') + '.pdf')

    # Second plot: sizes table
    table_string = table_function(sizes_arrays,labels)
    os.makedirs('../outputs',exist_ok=True)
    table_file = open('../' + (f'outputs/violins_tables/{alpha}_{delta}_coco_table').replace('.','_') + '.txt', "w")
    n = table_file.write(table_string)
    table_file.close()

def experiment(rejection_region_functions,rejection_region_names,alpha,delta,num_lam,num_calib,lambdas_example_table,num_trials,coco_val_2017_directory,coco_instances_val_2017_json):
    df_list = []

    for idx in range(len(rejection_region_functions)):
        rejection_region_function = rejection_region_functions[idx]
        rejection_region_name = rejection_region_names[idx]
        print(rejection_region_name)
        fname = f'../.cache/{alpha}_{delta}_{num_lam}_{num_calib}_{num_trials}_{rejection_region_name}_dataframe.pkl'

        df = pd.DataFrame(columns = ["$\\hat{\\lambda}$","FDR","sizes","alpha","delta","region name"])
        try:
            df = pd.read_pickle(fname)
            df["region name"] = rejection_region_name
            df.to_pickle(fname)
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
        fdr_median = np.median(df["FDR"])
        lambda_median = np.median(df["$\\hat{\\lambda}$"])
        fraction_violations = np.mean(df["FDR"] > alpha)
        print(f"alpha:{alpha}, method:{rejection_region_name}, median fdr: {fdr_median}, median lambda:{lambda_median}, fraction violations: {fraction_violations}")

    plot_histograms(df_list,alpha,delta)


if __name__ == "__main__":
    with torch.no_grad():
        sns.set(palette='pastel',font='serif')
        sns.set_style('white')
        fix_randomness(seed=0)
        args = parse_args(parser)
        coco_val_2017_directory = '/home/group/coco/val2017'
        coco_instances_val_2017_json = '/home/group/coco/annotations/instances_val2017.json'

        alphas = [0.5,0.2,0.1,0.05]
        deltas = [0.1,0.1,0.1,0.1]
        params = list(zip(alphas,deltas))
        num_lam = 10000 
        num_calib = 4000 
        num_trials = 100 
        lambdas_example_table = np.linspace(0,1,num_lam)

        # local function to preserve template
        def _bonferroni_search_HB_J1(loss_table,lambdas,alpha,delta):
            return bonferroni_search_HB(loss_table,lambdas,alpha,delta,downsample_factor=loss_table.shape[1])

        # local function to preserve template
        def _bonferroni_search_HB(loss_table,lambdas,alpha,delta):
            return bonferroni_search_HB(loss_table,lambdas,alpha,delta,downsample_factor=10)

        # local function to preserve template
        def _multiscale_bonferroni_search_HB(loss_table,lambdas,alpha,delta):
            return multiscale_bonferroni_search_HB(loss_table,lambdas,alpha,delta,downsample_factor=loss_table.shape[1])

        rejection_region_functions = ( uniform_region, bonferroni_HB, _bonferroni_search_HB, _bonferroni_search_HB_J1 )
        #rejection_region_functions = ( uniform_region, bonferroni_HB, _bonferroni_search_HB, _bonferroni_search_HB_J1, romano_wolf_multiplier_bootstrap )
        rejection_region_names = ( 'Uniform', 'Bonferroni', 'Fixed Sequence\n(Multi-Start)', 'Fixed Sequence' )
        #rejection_region_names = ( 'Uniform', 'Bonferroni', 'Fixed Sequence\n(Multi-Start)', 'Fixed Sequence', 'Multiplier\nBootstrap' )
        
        for alpha, delta in params:
            print(f"\n\n\n ============           NEW EXPERIMENT alpha={alpha} delta={delta}           ============ \n\n\n") 
            experiment(rejection_region_functions,rejection_region_names,alpha,delta,num_lam,num_calib,lambdas_example_table,num_trials,coco_val_2017_directory,coco_instances_val_2017_json)
