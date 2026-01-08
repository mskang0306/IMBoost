import os
os.getcwd()
os.chdir('/home/dongha0718/ms/IMBoost')

import logging
import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset  
from base.base_net import BaseNet
from sklearn.metrics import roc_auc_score
from PIL import Image
from datasets.mnist import Refine_MNIST_Dataset
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from networks.main import build_network



def binarize(inputs):
    inputs[inputs > 0.5] = 1.
    inputs[inputs <= 0.5] = 0.
    return inputs


def IMBoost(dataset_name, dataset, filter_net_name, model_seed, logger, train_option, gamma_, qt_, train_n, query_strategy='RD', lambda_1=2.0, lambda_2=1.0, xi=0.4, MM_alpha=0.4, pre_epoch=50, fine_epoch=50):

    if 'VAE_alpha1.' in train_option:
        num_sam = 1
        num_aggr = 1
        alpha = 1.
    elif 'IWAE_alpha1.' in train_option:
        #num_sam = 100
        #num_aggr = 50
        # For CNN
        num_sam = 2
        num_aggr = 1
        alpha = 1.
    elif 'VAE_alpha100.' in train_option:
        num_sam = 1
        num_aggr = 1
        alpha = 100.
    elif 'IWAE_alpha100.' in train_option:
        num_sam = 100
        num_aggr = 50
        alpha = 100.
    elif 'VAE_alpha0.01' in train_option:
        num_sam = 1
        num_aggr = 1
        alpha = 0.01
    elif 'IWAE_alpha0.01' in train_option:
        num_sam = 100
        num_aggr = 50
        alpha = 0.01
    
    
    import random
    lr_milestone = 50
    weight_decay = 0.5e-6
    device = 'cuda'

    filter_model_lr = 1e-3
    pretrain_epochs = pre_epoch
    finetune_epochs = fine_epoch
    tot_filter_model_n_epoch = pretrain_epochs + finetune_epochs

    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed(model_seed)
    torch.backends.cudnn.deterministic = True


    # 1) Labeled data before training
    n_jobs = 0
    label_bs = 128
    label_loader, _ = dataset.loaders(batch_size=label_bs, shuffle_train=False, drop_last=False, num_workers=n_jobs)

    all_inputs, all_targets, all_indices = [], [], []
    for inputs, targets, idxs in label_loader:
        all_inputs.append(inputs)
        all_targets.append(targets)
        all_indices.append(idxs)
    all_inputs = torch.cat(all_inputs)
    all_targets = torch.cat(all_targets).long()
    all_indices = torch.cat(all_indices)

    # 2) Active Learning inital masking
    known_ratio = 0.0  
    masked_targets, original_targets = mask_labels_for_active_learning(all_targets, known_ratio)


    train_interval_loss = []
    train_interval_targets = []
    train_interval_idxs = []

    known_train_iter = None


    # Initialize DeepSAD model and set neural network phi
    filter_model = build_network(filter_net_name)
    filter_model = filter_model.to(device)
    filter_optimizer = optim.Adam(filter_model.parameters(), lr=filter_model_lr, weight_decay=weight_decay)

    # Training
    logger.info('Starting train filter_model...')
    n_jobs_dataloader = 0
    train_iteration = 5
    m_0 = 128
    gamma = gamma_ 
    qt = qt_
    m_1 = qt * m_0
    epoch_qt = 10

        
    for epoch in range(tot_filter_model_n_epoch):
        # warm_up stage
        if epoch<epoch_qt:
            adaptive_batch_size = m_0

        # Applying adaptive batch size
        else: 
            if 'CIFAR'in dataset_name:
                if gamma_ > 1.01:
                    adaptive_batch_size = min(int(m_0 * (gamma ** (epoch-epoch_qt))), train_n, 3000)
            elif 'MNIST-C'in dataset_name:
                if gamma_ > 1.01:
                    adaptive_batch_size = min(int(m_0 * (gamma ** (epoch-epoch_qt))), train_n, 1000)
            elif 'InternetAds' in dataset_name:
                if gamma_ > 1.01:
                    adaptive_batch_size = min(int(m_0 * (gamma ** (epoch-epoch_qt))), train_n, 1000)
            elif 'SVHN' in dataset_name:
                if gamma_ > 1.01:
                    adaptive_batch_size = min(int(m_0 * (gamma ** (epoch-epoch_qt))), train_n, 1000)
            elif dataset_name=='mnist':
                if gamma_ > 1.01:
                    adaptive_batch_size = min(int(m_0 * (gamma ** (epoch-epoch_qt))), train_n, 5000)
            else:
                adaptive_batch_size = min(int(m_0 * (gamma ** (epoch-epoch_qt))), train_n, 20000)


        train_loader, _ = dataset.loaders(batch_size=adaptive_batch_size, drop_last=True, num_workers=n_jobs_dataloader)
 
        _, test_loader = dataset.loaders(batch_size=128, drop_last=False, num_workers=n_jobs_dataloader)
  
        train_loader_noshuf, _  = dataset.loaders(batch_size=128, shuffle_train=False, drop_last = False, num_workers=n_jobs_dataloader)


        # Setting of train_loader
        if epoch < pretrain_epochs:
            train_iter = iter(train_loader)
        else:
            train_iter = iter(unknown_train_loader)


        ### Train VAE
        for batch_idx in range(train_iteration):
            
            try:
                inputs, targets, idx = next(train_iter)
            except StopIteration:
                if epoch < pretrain_epochs:
                    train_iter = iter(train_loader)
                else:
                    train_iter = iter(unknown_train_loader)
                try:
                    inputs, targets, idx = next(train_iter)
                except StopIteration:
                    continue

            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(inputs.size(0), -1)


            if (epoch >= pretrain_epochs): 
                try:
                    known_inputs, known_targets, _ = next(known_train_iter)
                except StopIteration:
                    known_train_iter = iter(known_train_loader)
                    known_inputs, known_targets, _ = next(known_train_iter)

                known_inputs, known_targets = known_inputs.to(device), known_targets.to(device)
                known_inputs = known_inputs.view(known_inputs.size(0), -1)


            #break
            if 'binarize' in train_option:
                inputs = binarize(inputs)
            # Zero the network parameter gradients
            filter_model.train()

            if inputs.size(0) == 1:
                filter_model.eval()

            filter_optimizer.zero_grad()
            # Update network parameters via backpropagation: forward + backward + optimize
            if 'pixel' in filter_net_name:
                loss,loss_vec = VAE_IWAE_loss_gaussian_mean_var_pixelcnn(inputs, filter_model, num_sam, num_aggr, alpha)
            elif 'gaussian' in filter_net_name:
                loss,loss_vec = VAE_IWAE_loss_gaussian_mean_var(inputs, filter_model, num_sam, num_aggr, alpha)
            elif 'gaussian' in train_option:
                loss,loss_vec = VAE_IWAE_loss_gaussian(inputs, filter_model, num_sam, num_aggr, alpha)
            else:
                loss,loss_vec = VAE_IWAE_loss(inputs, filter_model, num_sam, num_aggr, alpha)
            filter_model.train()


            #####################################################################
            #####################################################################

            inlier_loss = torch.tensor(0.0, device=inputs.device)
            outlier_cubo_loss = torch.tensor(0.0, device=inputs.device)

            if (epoch >= pretrain_epochs): 

                # Inlier Loss 
                inlier_inputs = known_inputs[known_targets == 0]
                if (len(inlier_inputs) > 0):
                    filter_model.eval()
                    if 'pixel' in filter_net_name:
                        inlier_loss = VAE_IWAE_loss_gaussian_mean_var_pixelcnn(inlier_inputs, filter_model, num_sam, num_aggr, alpha)[0]
                    elif 'gaussian' in filter_net_name:
                        inlier_loss = VAE_IWAE_loss_gaussian_mean_var(inlier_inputs, filter_model, num_sam, num_aggr, alpha)[0]
                    elif 'gaussian' in train_option:
                        inlier_loss = VAE_IWAE_loss_gaussian(inlier_inputs, filter_model, num_sam, num_aggr, alpha)[0]
                    else:
                        inlier_loss = VAE_IWAE_loss(inlier_inputs, filter_model, num_sam, num_aggr, alpha)[0]
                    filter_model.train()
                else:
                    inlier_loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)

                # Outlier CUBO Loss 
                outlier_inputs = known_inputs[known_targets == 1]
                if len(outlier_inputs) > 0: 
                    filter_model.eval() 
                    if 'pixel' in filter_net_name:
                        outlier_cubo_loss = VAE_CUBO_loss_gaussian_mean_var_pixelcnn(outlier_inputs, filter_model, num_sam, num_aggr, u=2.0)[0]
                    elif 'gaussian' in filter_net_name:
                        outlier_cubo_loss = VAE_CUBO_loss_gaussian_mean_var(outlier_inputs, filter_model, num_sam, num_aggr, u=2.0)[0]
                    elif 'gaussian' in train_option:
                        outlier_cubo_loss = VAE_CUBO_loss_gaussian(outlier_inputs, filter_model, num_sam, num_aggr, u=2.0)[0]
                    else:
                        outlier_cubo_loss = VAE_CUBO_loss(outlier_inputs, filter_model, num_sam, num_aggr, u=2.0)[0]
                    filter_model.train()  
                else:
                    outlier_cubo_loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)

            # Loss 
            loss_quantile_prob = 1 - (np.clip((epoch - epoch_qt), 0, 1) * (1 - m_1 / m_0))
            loss_quantile = torch.quantile(loss_vec.detach(), loss_quantile_prob)
            tau = xi * inlier_loss + (1 - xi) * loss_quantile 
            trimming_ind = (loss_vec.detach() <= tau)

            if trimming_ind.sum() == 0:
                trimmed_loss = torch.tensor(0.0, device=loss_vec.device, requires_grad=True)
            else:
                trimmed_loss = (loss_vec * trimming_ind).sum() / trimming_ind.sum()

            # lambda 
            if epoch < pretrain_epochs:
                ep_lambda1 = 0.0
                ep_lambda2 = 0.0
            else:
                ep_lambda1 = lambda_1
                ep_lambda2 = lambda_2

            # Total Loss
            total_loss = trimmed_loss + ep_lambda1 * inlier_loss + ep_lambda2 * outlier_cubo_loss

            total_loss.backward()
            filter_optimizer.step()



        if epoch >= pretrain_epochs - 10:                
            train_loss_list_eval = []
            train_targets_list_eval = []
            train_idx_list_eval = []

            filter_model.eval()
            for data in train_loader_noshuf:
                inputs, targets, idx = data
                inputs = inputs.to(device)
                inputs = inputs.view(inputs.size(0), -1)
                if 'binarize' in train_option:
                    inputs = binarize(inputs)
                # Update network parameters via backpropagation: forward + backward + optimize
                if 'pixel' in filter_net_name:
                    loss,loss_vec = VAE_IWAE_loss_gaussian_mean_var_pixelcnn(inputs, filter_model, num_sam, num_aggr, 1.)
                elif 'gaussian' in filter_net_name:
                    loss,loss_vec = VAE_IWAE_loss_gaussian_mean_var(inputs, filter_model, num_sam, num_aggr, 1.)
                elif 'gaussian' in train_option:
                    loss,loss_vec = VAE_IWAE_loss_gaussian(inputs, filter_model, num_sam, num_aggr, 1.)
                else:
                    loss,loss_vec = VAE_IWAE_loss(inputs, filter_model, num_sam, num_aggr, 1.)

                train_loss_list_eval.append(loss_vec.data.cpu())
                train_targets_list_eval.append(targets.cpu().numpy())
                train_idx_list_eval += list(idx.numpy())

            loss_np = torch.cat(train_loss_list_eval, dim=0).numpy().reshape(-1, 1)
            targets_np = np.concatenate(train_targets_list_eval)
            idxs_np = np.array(train_idx_list_eval)

            train_interval_loss.append(loss_np)
            train_interval_targets.append(targets_np)
            train_interval_idxs.append(idxs_np)

            
            # Active Learning
            if epoch % 10 == 9:
                avg_train_loss = np.mean(np.stack(train_interval_loss, axis=0), axis=0).squeeze()
                avg_targets = train_interval_targets[-1]
                avg_idxs = train_interval_idxs[-1]

                if epoch != tot_filter_model_n_epoch - 1:
                    candidate_indices = get_candidate_indices(masked_targets)
                    initial_total = len(all_targets)
                    query_size = max(1, int(initial_total * 0.01)) if initial_total > 500 else min(6, len(candidate_indices))

                    if query_strategy == 'RD':
                        selected_indices = RD(candidate_indices, query_size)
                    elif query_strategy == 'MM':
                        selected_indices = MM(candidate_indices, avg_train_loss, query_size, MM_alpha=MM_alpha)
                    elif query_strategy == 'CP':
                        selected_indices = CP(candidate_indices, avg_train_loss, query_size, alpha=0.5)
                    
                    # label, loader update
                    masked_targets = update_labels_with_active_learning(selected_indices, masked_targets, original_targets)

                    known_mask = (masked_targets != -1)
                    unknown_mask = (masked_targets == -1)

                    # known_train_loader 
                    known_inputs = all_inputs[known_mask]
                    known_targets = masked_targets[known_mask]
                    known_idxs = all_indices[known_mask]
                    known_train_dataset = torch.utils.data.TensorDataset(known_inputs, known_targets, known_idxs)
                    known_train_loader = torch.utils.data.DataLoader(known_train_dataset, batch_size = min(adaptive_batch_size, known_inputs.size(0)), shuffle=True)
                    known_train_iter = iter(known_train_loader)

                    # unknown_train_loader
                    unknown_inputs = all_inputs[unknown_mask]
                    unknown_targets = masked_targets[unknown_mask] 
                    unknown_idxs = all_indices[unknown_mask]
                    unknown_train_dataset = torch.utils.data.TensorDataset(unknown_inputs, unknown_targets, unknown_idxs)
                    unknown_train_loader = torch.utils.data.DataLoader(unknown_train_dataset, batch_size=adaptive_batch_size, shuffle=True)

                # interval initialization
                train_interval_loss.clear()
                train_interval_targets.clear()
                train_interval_idxs.clear()


            if epoch == tot_filter_model_n_epoch - 1:

                train_loss_np = torch.cat(train_loss_list_eval, dim=0).numpy()
                train_idxs_np = np.array(train_idx_list_eval)

                idx_mean_loss_train = pd.DataFrame({'idx': train_idxs_np, 'loss': train_loss_np})

        
        print(epoch)



    test_loss_list = []
    test_idx_list = []

    filter_model.eval()
    with torch.no_grad():
        for inputs, targets, idx in test_loader:
            inputs = inputs.to(device).view(inputs.size(0), -1)
            if 'binarize' in train_option:
                inputs = binarize(inputs)
            # Update network parameters via backpropagation: forward + backward + optimize
            if 'pixel' in filter_net_name:
                loss,loss_vec = VAE_IWAE_loss_gaussian_mean_var_pixelcnn(inputs , filter_model , num_sam , num_aggr,1.)
            elif 'gaussian' in filter_net_name:
                loss,loss_vec = VAE_IWAE_loss_gaussian_mean_var(inputs , filter_model , num_sam , num_aggr,1.)
            elif 'gaussian' in train_option:
                loss,loss_vec = VAE_IWAE_loss_gaussian(inputs , filter_model , num_sam , num_aggr,1.)
            else:
                loss,loss_vec = VAE_IWAE_loss(inputs , filter_model , num_sam , num_aggr,1.)

            test_loss_list.append(loss_vec.data.cpu())
            test_idx_list += list(idx.numpy())

    test_loss_np = torch.cat(test_loss_list, dim=0).numpy()
    test_idxs_np = np.array(test_idx_list)

    idx_mean_loss_test = pd.DataFrame({'idx': test_idxs_np, 'loss': test_loss_np})
    

    return idx_mean_loss_train, idx_mean_loss_test



################################################################################################
def logp_z_normal(z , z_mu , z_log_var):
    ## z : mini_batch * z_dim
    ## z_mu , z_log_var : mini_batch * z_dim
    
    logp = -z_log_var/2-torch.pow((z-z_mu) , 2)/(2*torch.exp(z_log_var))
    
    return logp.sum(1)  ## (mini_batch,) vector

################################################################################################
## Calculate log p(z)
################################################################################################
def logp_z(z , z_mu_ps , z_log_var_ps , p_cluster):
    ## z : mini_batch * z_dim
    ## z_mu , z_log_var : n_pseudo * z_dim
    z_ten = z.unsqueeze(1)                                                              ## mini_batch * 1 * z_dim
    z_mu_ps_ten , z_log_var_ps_ten = z_mu_ps.unsqueeze(0) , z_log_var_ps.unsqueeze(0)   ## 1 * n_pseudo * z_dim
    
    logp = (-z_log_var_ps_ten/2-torch.pow((z_ten-z_mu_ps_ten) , 2)/(2*torch.exp(z_log_var_ps_ten))).sum(2) + \
            (torch.log(p_cluster).unsqueeze(0)) ## logp : mini_batch * n_pseudo
    
    max_logp = torch.max(logp , 1)[0]
    #logp = torch.log((torch.exp(logp)).sum(1))
    
    logp = max_logp + torch.log(torch.sum(torch.exp(logp - max_logp.unsqueeze(1)), 1))
    
    return logp

################################################################################################
## Calculate log p(z|x)
################################################################################################
def logp_z_given_x(z , z_mu , z_log_var):
    ## z : mini_batch * z_dim
    ## z_mu , z_log_var : mini_batch * z_dim    
    logp = -z_log_var/2-torch.pow((z-z_mu) , 2)/(2*torch.exp(z_log_var))
    
    return logp.sum(1)  ## (mini_batch,) vector

################################################################################################
## Calculate log p(x|z)
################################################################################################
def logp_x_given_z(x , x_mu):
    ## x : mini_batch * x_dim
    ## x_mu : mini_batch * x_dim

    eps = 1e-5
    logp = (x * torch.log(x_mu+eps) + (1. - x) * torch.log(1.-x_mu+eps)).sum(1)

    #logp = -(1/2)*((x-x_mu)**2).sum(1)
    
    return logp

################################################################################################
## Calculate log p(x|z)
################################################################################################
def logp_x_given_z_gaussian(x , x_mu):
    ## x : mini_batch * x_dim
    ## x_mu : mini_batch * x_dim

    eps = 1e-5
    #logp = (x * torch.log(x_mu+eps) + (1. - x) * torch.log(1.-x_mu+eps)).sum(1)

    logp = -(1/2)*((x-x_mu)**2).sum(1)
    
    return logp

################################################################################################
## Calculate log p(x|z)
################################################################################################
def logp_x_given_z_gaussian_mean_var(x , x_mu, x_logvar):
    ## x : mini_batch * x_dim
    ## x_mu : mini_batch * x_dim

    eps = 1e-5
    #logp = (x * torch.log(x_mu+eps) + (1. - x) * torch.log(1.-x_mu+eps)).sum(1)

    #logp = -(1/2)*((x-x_mu)**2).sum(1)
    logp = (-x_logvar-(1/(2*torch.exp(x_logvar)))*((x-x_mu)**2)).sum(1)
    
    return logp

################################################################################################
## Calculate log pdf of normal distribution
## z ~ N(0,1)
################################################################################################
def logp_z_std_mvn(z):
    ## z : mini_batch * z_dim
    ## z_mu , z_log_var : mini_batch * z_dim
    
    logp = -torch.pow(z , 2)/2
    
    return logp.sum(1)  ## (mini_batch,) vector

################################################################################################
## Calculate log pdf of normal distribution
################################################################################################
def logq_z(z , z_mu , z_log_var):
    ## z : mini_batch * z_dim
    ## z_mu , z_log_var : mini_batch * z_dim
    
    logp = -z_log_var/2-torch.pow((z-z_mu) , 2)/(2*torch.exp(z_log_var))
    
    return logp.sum(1)  ## (mini_batch,) vector

################################################################################################
## Calculate calculate_center function
################################################################################################
def calculate_center(x , gmvae):   

    z_mu , z_log_var = gmvae.encoder(x)
        
    return z_mu

################################################################################################
## Calculate IWAE loss function
################################################################################################
def VAE_IWAE_loss(x , gmvae , num_sam , num_aggr,alpha):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    x=x.repeat(num_aggr,1)
    
    z_mu , z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z = gmvae.decoder(sam_z)
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z_std_mvn(sam_z)
        log_q = logp_z_normal(sam_z , z_mu , z_log_var)
        log_recon = logp_x_given_z(x , x_mu_sam_z)


        elbo = log_recon + alpha*(log_p - log_q)
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())
        

            
    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)
            
    return -log_loss.mean(),-log_loss

################################################################################################
## Calculate IWAE loss function
################################################################################################
def VAE_IWAE_loss_gaussian(x , gmvae , num_sam , num_aggr,alpha):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    x=x.repeat(num_aggr,1)
    
    z_mu , z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z = gmvae.decoder(sam_z)
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z_std_mvn(sam_z)
        log_q = logp_z_normal(sam_z , z_mu , z_log_var)
        log_recon = logp_x_given_z_gaussian(x , x_mu_sam_z)


        elbo = log_recon + alpha*(log_p - log_q)
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())
        

            
    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)
            
    return -log_loss.mean(),-log_loss

################################################################################################
## Calculate IWAE loss function
################################################################################################
def VAE_IWAE_loss_gaussian_mean_var_pixelcnn(x , gmvae , num_sam , num_aggr,alpha):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    x=x.repeat(num_aggr,1)
    
    z_mu , z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z,x_logvar_sam_z = gmvae.decoder(x,sam_z)
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        x_logvar_sam_z = x_logvar_sam_z.view(x_logvar_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z_std_mvn(sam_z)
        log_q = logp_z_normal(sam_z , z_mu , z_log_var)
        log_recon = logp_x_given_z_gaussian_mean_var(x , x_mu_sam_z, x_logvar_sam_z)


        elbo = log_recon + alpha*(log_p - log_q)
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())
        

            
    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)
            
    return -log_loss.mean(),-log_loss

################################################################################################
## Calculate IWAE loss function
################################################################################################
def VAE_IWAE_loss_gaussian_mean_var(x , gmvae , num_sam , num_aggr,alpha):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    x=x.repeat(num_aggr,1)
    
    z_mu , z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z,x_logvar_sam_z = gmvae.decoder(sam_z)
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        x_logvar_sam_z = x_logvar_sam_z.view(x_logvar_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z_std_mvn(sam_z)
        log_q = logp_z_normal(sam_z , z_mu , z_log_var)
        log_recon = logp_x_given_z_gaussian_mean_var(x , x_mu_sam_z, x_logvar_sam_z)


        elbo = log_recon + alpha*(log_p - log_q)
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())
        

            
    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)
            
    return -log_loss.mean(),-log_loss

################################################################################################
## Calculate CUBO loss function
################################################################################################
def VAE_CUBO_loss(x, gmvae, num_sam, num_aggr, u=2.0):    

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    x=x.repeat(num_aggr, 1)
    
    z_mu, z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z, _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z = gmvae.decoder(sam_z)
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z_std_mvn(sam_z)
        log_q = logp_z_normal(sam_z, z_mu, z_log_var)
        log_recon = logp_x_given_z(x, x_mu_sam_z)

        elbo_sq = u * (log_recon + log_p - log_q)
        log_loss_list.append(elbo_sq.reshape(num_aggr , -1).t())
                   
    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    cubo_loss = (torch.log(log_loss_list) + max_log_loss.squeeze(1)) / u
          
    return cubo_loss.mean(), cubo_loss

################################################################################################
## Calculate CUBO loss function 
################################################################################################
def VAE_CUBO_loss_gaussian(x, gmvae, num_sam, num_aggr, u=2.0):    

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    x=x.repeat(num_aggr, 1)
    
    z_mu, z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z, _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z = gmvae.decoder(sam_z)
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z_std_mvn(sam_z)
        log_q = logp_z_normal(sam_z , z_mu , z_log_var)
        log_recon = logp_x_given_z_gaussian(x , x_mu_sam_z)

        elbo_sq = u * (log_recon + log_p - log_q)
        log_loss_list.append(elbo_sq.reshape(num_aggr , -1).t())
                   
    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    cubo_loss = (torch.log(log_loss_list) + max_log_loss.squeeze(1)) / u
          
    return cubo_loss.mean(), cubo_loss

################################################################################################
## Calculate CUBO loss function 
################################################################################################
def VAE_CUBO_loss_gaussian_mean_var_pixelcnn(x, gmvae, num_sam, num_aggr, u=2.0):    

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    x=x.repeat(num_aggr, 1)
    
    z_mu, z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z,x_logvar_sam_z = gmvae.decoder(x,sam_z)
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        x_logvar_sam_z = x_logvar_sam_z.view(x_logvar_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z_std_mvn(sam_z)
        log_q = logp_z_normal(sam_z , z_mu , z_log_var)
        log_recon = logp_x_given_z_gaussian_mean_var(x , x_mu_sam_z, x_logvar_sam_z)

        elbo_sq = u * (log_recon + log_p - log_q)
        log_loss_list.append(elbo_sq.reshape(num_aggr , -1).t())
                   
    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    cubo_loss = (torch.log(log_loss_list) + max_log_loss.squeeze(1)) / u
          
    return cubo_loss.mean(), cubo_loss

################################################################################################
## Calculate CUBO loss function 
################################################################################################
def VAE_CUBO_loss_gaussian_mean_var(x, gmvae, num_sam, num_aggr, u=2.0):    

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    x=x.repeat(num_aggr, 1)
    
    z_mu, z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z,x_logvar_sam_z = gmvae.decoder(sam_z)
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        x_logvar_sam_z = x_logvar_sam_z.view(x_logvar_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z_std_mvn(sam_z)
        log_q = logp_z_normal(sam_z , z_mu , z_log_var)
        log_recon = logp_x_given_z_gaussian_mean_var(x , x_mu_sam_z, x_logvar_sam_z)

        elbo_sq = u * (log_recon + log_p - log_q)
        log_loss_list.append(elbo_sq.reshape(num_aggr , -1).t())
                   
    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    cubo_loss = (torch.log(log_loss_list) + max_log_loss.squeeze(1)) / u
          
    return cubo_loss.mean(), cubo_loss



################################################################################################
## Active Learning
################################################################################################

def mask_labels_for_active_learning(targets, known_ratio):

    targets = targets.clone()
    original_targets = targets.clone()
    
    num_samples = len(targets)
    known_samples = int(num_samples * known_ratio)
    
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    known_indices = indices[:known_samples]
    unknown_indices = indices[known_samples:]
    targets[unknown_indices] = -1

    return targets, original_targets


def update_labels_with_active_learning(selected_indices, masked_targets, original_targets):

    masked_targets = masked_targets.clone()
    masked_targets[selected_indices] = original_targets[selected_indices]
    return masked_targets


def get_candidate_indices(masked_targets):

    return np.where(masked_targets == -1)[0]



################################################################################################
## Query Strategy
################################################################################################

def RD(candidate_indices, num_samples):

    if len(candidate_indices) == 0:
        return np.array([])

    num_select = min(len(candidate_indices), num_samples)
    selected_indices = np.random.choice(candidate_indices, size=num_select, replace=False)

    return selected_indices


def MM(candidate_indices, loss_vec, num_samples, MM_alpha):

    if len(candidate_indices) == 0:
        return np.array([])

    from sklearn.mixture import GaussianMixture

    # GMM
    gmm = GaussianMixture(n_components=2, random_state=0)
    loss_reshaped = loss_vec.reshape(-1, 1)
    gmm.fit(loss_reshaped)

    probs = gmm.predict_proba(loss_reshaped)
    posterior_min = np.min(probs, axis=1)

    candidate_scores = posterior_min[candidate_indices]

    distance_to_MM_alpha = np.abs(candidate_scores - MM_alpha)
    sorted_indices = np.argsort(distance_to_MM_alpha)

    top_k_indices = sorted_indices[:num_samples]
    selected_indices = candidate_indices[top_k_indices]

    return selected_indices


def CP(candidate_indices, loss_vec, num_samples, alpha=0.5):

    if len(candidate_indices) == 0:
        return np.array([])

    candidate_losses = loss_vec[candidate_indices]
    sorted_indices = np.argsort(candidate_losses)

    num_low = int(num_samples * alpha)
    num_high = num_samples - num_low

    low_indices = candidate_indices[sorted_indices[:num_low]] if num_low > 0 else np.array([], dtype=int)
    high_indices = candidate_indices[sorted_indices[-num_high:]] if num_high > 0 else np.array([], dtype=int)

    selected_indices = np.concatenate([low_indices, high_indices])

    return selected_indices
