# -*- coding: utf-8 -*-
"""
Created on 2026-02-09 (Mon) 17:35:33

Proposed Model Route2

@author: I.Azuma
"""
# %%
BASE_DIR = '/home/aah18044co/github'


import time 
import random
import argparse
import numpy as np
import pandas as pd

from sklearn import metrics 
from datetime import datetime
from collections import defaultdict

import torch
from torch import optim

import sys
sys.path.append(f'{BASE_DIR}/XAI-DDI/model')

#import os
#os.chdir(f'{BASE_DIR}/model/route2')

from route2 import models, custom_loss
from route2.data_preprocessing import DrugDataset, DrugDataLoader
import warnings
warnings.filterwarnings('ignore',category=UserWarning)

sys.path.append(f'{BASE_DIR}/wandb-util')
from wandbutil import WandbLogger


SEED=42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
g = torch.Generator()
g.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--n_atom_feats', type=int, default=55, help='num of input features')
parser.add_argument('--n_atom_hid', type=int, default=128, help='num of hidden features')
parser.add_argument('--rel_total', type=int, default=86, help='num of interaction types')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=200, help='num of epochs')
parser.add_argument('--kge_dim', type=int, default=128, help='dimension of interaction matrix')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')  # NOTE: memory error when 1024

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--neg_samples', type=int, default=1)
parser.add_argument('--data_size_ratio', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True, choices=[0, 1])
parser.add_argument('--pkl_name', type=str, default=f'{BASE_DIR}/XAI-DDI/workspace/abci_benchmarking/results/260218/selected_genes_attention.pkl')

# args = parser.parse_args()
args, unknown = parser.parse_known_args(args=[])  # for jupyter notebook
n_atom_feats = args.n_atom_feats
n_atom_hid = args.n_atom_hid
rel_total = args.rel_total
lr = args.lr
n_epochs = args.n_epochs
kge_dim = args.kge_dim
batch_size = args.batch_size
pkl_name = args.pkl_name

weight_decay = args.weight_decay
neg_samples = args.neg_samples
data_size_ratio = args.data_size_ratio
device = 'cuda:0' if torch.cuda.is_available() and args.use_cuda else 'cpu'
print(args)

# %% Dataset
df_ddi_train = pd.read_csv(f'{BASE_DIR}/XAI-DDI/dataset/inductive_data/fold1/train.csv')
df_ddi_s1 = pd.read_csv(f'{BASE_DIR}/XAI-DDI/dataset/inductive_data/fold1/s1.csv')
df_ddi_s2 = pd.read_csv(f'{BASE_DIR}/XAI-DDI/dataset/inductive_data/fold1/s2.csv')

train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
s1_tup = [(h, t, r) for h, t, r in zip(df_ddi_s1['d1'], df_ddi_s1['d2'], df_ddi_s1['type'])]
s2_tup = [(h, t, r) for h, t, r in zip(df_ddi_s2['d1'], df_ddi_s2['d2'], df_ddi_s2['type'])]

train_data = DrugDataset(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)
s1_data = DrugDataset(s1_tup, disjoint_split=True)
s2_data = DrugDataset(s2_tup, disjoint_split=True)

print(f"Training with {len(train_data)} samples, s1 with {len(s1_data)}, and s2 with {len(s2_data)}")

train_data_loader = DrugDataLoader(train_data, batch_size=batch_size,shuffle=True,num_workers=0)
s1_data_loader = DrugDataLoader(s1_data, batch_size=batch_size *3,num_workers=0)
s2_data_loader = DrugDataLoader(s2_data, batch_size=batch_size *3,num_workers=0)

# load KG embeddings
kg_features = pd.read_pickle(f'{BASE_DIR}/XAI-DDI/dataset/kg_embeddings/selected_genes_14662_embeddings.pkl')

# %%
def do_compute(batch, device, model, kg_features):
    '''
        *batch: (pos_tri, neg_tri)
        *pos/neg_tri: (batch_h, batch_t, batch_r)
    '''
    probas_pred, ground_truth = [], []
    pos_tri, neg_tri, original_index = batch
    
    # positive
    pos_tri = [tensor.to(device=device) for tensor in pos_tri]
    p_score, attention_weights = model(pos_tri, kg_features)
    probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
    ground_truth.append(np.ones(len(p_score)))

    # negative
    neg_tri = [tensor.to(device=device) for tensor in neg_tri]
    n_score, attention_weights = model(neg_tri, kg_features)
    probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
    ground_truth.append(np.zeros(len(n_score)))

    probas_pred = np.concatenate(probas_pred)
    ground_truth = np.concatenate(ground_truth)

    return p_score, n_score, probas_pred, ground_truth

def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, t = metrics.precision_recall_curve(target, probas_pred)
    int_ap = metrics.auc(r, p)
    ap= metrics.average_precision_score(target, probas_pred)

    return acc, auroc, f1_score, precision, recall, int_ap, ap


# %% Model preparation
model = models.MVN_DDI(n_atom_feats, n_atom_hid, kge_dim, rel_total, heads_out_feat_params=[64,64,64,64], blocks_params=[2, 2, 2, 2], kg_emb_dim=200)
loss_fn = custom_loss.SigmoidLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))
model.to(device=device)

option_list = defaultdict(list)
option_list['n_atom_feats'] = n_atom_feats
option_list['n_atom_hid'] = n_atom_hid
option_list['rel_total'] = rel_total
option_list['lr'] = lr
option_list['n_epochs'] = n_epochs
option_list['kge_dim'] = kge_dim
option_list['batch_size'] = batch_size
option_list['weight_decay'] = weight_decay
option_list['neg_samples'] = neg_samples
option_list['data_size_ratio'] = data_size_ratio

logger = WandbLogger(
    entity="XAI-DDI",  
    project="251101_route2_pairfeats_crossattn",  
    group="abci", 
    name="260218_selected_genes_attention",
    config=option_list,
)


# %% Train phase
print('Starting training at', datetime.today())
s1_acc_max = 0
s2_acc_max = 0
for epoch in range(1, n_epochs+1):
    start = time.time()
    train_loss = 0 
    s1_loss = 0
    s2_loss = 0
    
    train_probas_pred = []
    train_ground_truth = []

    s1_probas_pred = []
    s1_ground_truth = []

    s2_probas_pred = []
    s2_ground_truth = []

    for batch in train_data_loader:
        model.train()

        p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model, kg_features)


        train_probas_pred.append(probas_pred)
        train_ground_truth.append(ground_truth)
        loss, loss_p, loss_n = loss_fn(p_score, n_score)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(p_score)
    train_loss /= len(train_data)

    with torch.no_grad():
        train_probas_pred = np.concatenate(train_probas_pred)
        train_ground_truth = np.concatenate(train_ground_truth)

        train_acc, train_auc_roc, train_f1, train_precision,train_recall,train_int_ap, train_ap = do_compute_metrics(train_probas_pred, train_ground_truth)

        # S1 scenario
        for batch in s1_data_loader:
            model.eval()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model, kg_features)
            s1_probas_pred.append(probas_pred)
            s1_ground_truth.append(ground_truth)
            loss, loss_p, loss_n = loss_fn(p_score, n_score)
            s1_loss += loss.item() * len(p_score)            

        s1_loss /= len(s1_data)
        s1_probas_pred = np.concatenate(s1_probas_pred)
        s1_ground_truth = np.concatenate(s1_ground_truth)
        s1_acc, s1_auc_roc, s1_f1,s1_precision,s1_recall,s1_int_ap, s1_ap = do_compute_metrics(s1_probas_pred, s1_ground_truth)
        
        # S2 scenario
        for batch in s2_data_loader:
            model.eval()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model, kg_features)
            s2_probas_pred.append(probas_pred)
            s2_ground_truth.append(ground_truth)
            loss, loss_p, loss_n = loss_fn(p_score, n_score)
            s2_loss += loss.item() * len(p_score)            

        s2_loss /= len(s2_data)
        s2_probas_pred = np.concatenate(s2_probas_pred)
        s2_ground_truth = np.concatenate(s2_ground_truth)
        s2_acc, s2_auc_roc, s2_f1,s2_precision,s2_recall,s2_int_ap, s2_ap = do_compute_metrics(s2_probas_pred, s2_ground_truth)

        if s1_acc>s1_acc_max:
            s1_acc_max = s1_acc
            torch.save(model,pkl_name)
            
    if scheduler:
        scheduler.step()
    
    loss_dict = {
        'train_loss': train_loss,
        's1_loss': s1_loss,
        's2_loss': s2_loss,
        'train_acc': train_acc,
        'train_roc': train_auc_roc,
        'train_precision': train_precision,
        'train_recall': train_recall,
        's1_acc': s1_acc,
        's1_roc': s1_auc_roc,
        's1_precision': s1_precision,
        's1_recall': s1_recall,
        's2_acc': s2_acc,
        's2_roc': s2_auc_roc,
        's2_precision': s2_precision,
        's2_recall': s2_recall,
    }
    logger(epoch=epoch, **loss_dict)
    
    print(f'Epoch: {epoch} ({time.time() - start:.4f}s), train_loss: {train_loss:.4f}, s1_loss: {s1_loss:.4f},s2_loss: {s2_loss:.4f}')
    print(f'\t\ttrain_acc: {train_acc:.4f}, train_roc: {train_auc_roc:.4f},train_precision: {train_precision:.4f},train_recall:{train_recall:.4f}')
    print(f'\t\ts1_acc: {s1_acc:.4f}, s1_roc: {s1_auc_roc:.4f}, s1_precision: {s1_precision:.4f}, s1_recall: {s1_recall:.4f}')
    print(f'\t\ts2_acc: {s2_acc:.4f}, s2_roc: {s2_auc_roc:.4f}, s2_precision: {s2_precision:.4f}, s2_recall: {s2_recall:.4f}')

    # clear cache
    torch.cuda.empty_cache()

# %% Test phase
test_model = torch.load(f'{BASE_DIR}/workspace/Model_Dev/route3/results/251205/selected_genes_attention.pkl')
s1_probas_pred = []
s1_ground_truth = []
s2_probas_pred = []
s2_ground_truth = []
with torch.no_grad():
    for batch in s1_data_loader:
        test_model.eval()
        p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, test_model, kg_features=kg_features)
        s1_probas_pred.append(probas_pred)
        s1_ground_truth.append(ground_truth)
    
    s1_probas_pred = np.concatenate(s1_probas_pred)
    s1_ground_truth = np.concatenate(s1_ground_truth)
    s1_acc, s1_auc_roc, s1_f1,s1_precision,s1_recall,s1_int_ap, s1_ap = do_compute_metrics(s1_probas_pred, s1_ground_truth)
    

    for batch in s2_data_loader:
        test_model.eval()
        p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, test_model, kg_features=kg_features)
        s2_probas_pred.append(probas_pred)
        s2_ground_truth.append(ground_truth)
            
    s2_probas_pred = np.concatenate(s2_probas_pred)
    s2_ground_truth = np.concatenate(s2_ground_truth)
    s2_acc, s2_auc_roc, s2_f1,s2_precision,s2_recall,s2_int_ap, s2_ap = do_compute_metrics(s2_probas_pred, s2_ground_truth)

print('\n')
print('============================== Best Result ==============================')
print(f'\t\ts1_acc: {s1_acc:.4f}, s1_roc: {s1_auc_roc:.4f}, s1_f1: {s1_f1:.4f}, s1_precision: {s1_precision:.4f},s1_recall: {s1_recall:.4f},s1_int_ap: {s1_int_ap:.4f},s1_ap: {s1_ap:.4f}')
print(f'\t\ts2_acc: {s2_acc:.4f}, s2_roc: {s2_auc_roc:.4f}, s2_f1: {s2_f1:.4f}, s2_precision: {s2_precision:.4f},s2_recall: {s2_recall:.4f},s2_int_ap: {s2_int_ap:.4f},s2_ap: {s2_ap:.4f}')

"""
251208結果
s1_acc: 0.6971, s1_roc: 0.7761, s1_f1: 0.6625, s1_precision: 0.7481,s1_recall: 0.5944,s1_int_ap: 0.7670,s1_ap: 0.7671
s2_acc: 0.7628, s2_roc: 0.8462, s2_f1: 0.7568, s2_precision: 0.7764,s2_recall: 0.7381,s2_int_ap: 0.8320,s2_ap: 0.8320
"""
