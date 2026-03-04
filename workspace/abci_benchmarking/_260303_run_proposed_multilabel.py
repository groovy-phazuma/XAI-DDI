#!/usr/bin/env python3
"""
Created on 2026-03-03 (Tue) 18:28:17

drugbankのDDIラベルを6クラスに集約してマルチラベル化する

データセットの調査: 260303_drugbank_multilabel_curation.py

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/HDD/Azuma_DDI'


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
sys.path.append(f'{BASE_DIR}/github/XAI-DDI/model')

import os
os.chdir(f'{BASE_DIR}/github/XAI-DDI')

from model.multi import models, custom_loss
from model.multi.data_preprocessing import DrugDataset, DrugDataLoader
import warnings
warnings.filterwarnings('ignore',category=UserWarning)

sys.path.append(f'{BASE_DIR}/github/wandb-util')
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
parser.add_argument('--pkl_name', type=str, default=f'{BASE_DIR}/workspace/Model_Dev/multi_label/results/260303/selected_genes_attention.pkl')

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
df_ddi_train = pd.read_csv(f'{BASE_DIR}/github/XAI-DDI/dataset/inductive_data/fold1/train.csv')
df_ddi_s1 = pd.read_csv(f'{BASE_DIR}/github/XAI-DDI/dataset/inductive_data/fold1/s1.csv')
df_ddi_s2 = pd.read_csv(f'{BASE_DIR}/github/XAI-DDI/dataset/inductive_data/fold1/s2.csv')

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
kg_features = pd.read_pickle(f'{BASE_DIR}/KG_datasource/MyKG/v0/embed/drkg_entity_emb_62507x200.pkl')
info_df = pd.read_csv(f'{BASE_DIR}/KG_datasource/MyKG/v0/entities.tsv', sep="\t", header=None)
info_df['type'] = info_df[0].apply(lambda x: x.split(':')[0])


selected_genes = pd.read_pickle(f'{BASE_DIR}/KG_datasource/MyKG/v0/251205_compound_neighbor_genes_14662.pkl')
selected_genes = sorted(list(selected_genes))
selected_indices = info_df[info_df[0].isin(selected_genes)].index.tolist()
kg_features = kg_features[selected_indices]
kg_features = torch.tensor(kg_features, dtype=torch.float32).to(device)

# %%
ddi_id_to_class = {}
# 変換辞書の作成
class_mapping = {
    0: [3, 4, 11, 13, 18, 43, 47, 62, 65, 67, 72, 73, 75, 77], # Pharmacokinetics
    1: [8, 45, 70], # Efficacy
    2: [5, 6, 10, 12, 15, 19, 20, 25, 26, 30, 31, 33, 37, 38, 50, 53, 54, 56, 58, 60, 63, 66, 71, 81, 82, 84, 85], # Cardio
    3: [2, 16, 17, 21, 22, 23, 27, 32, 35, 36, 40, 41, 44, 61, 64, 76, 79], # Nervous
    4: [9, 24, 29, 42, 51, 55, 68, 74, 78, 83], # Metabolic
    5: [1, 7, 14, 28, 34, 39, 46, 48, 49, 52, 57, 59, 69, 80, 86] # Others
}
ddi_id_to_class = {ddi_id: cls_id for cls_id, ids in class_mapping.items() for ddi_id in ids}

for class_id, ddi_ids in class_mapping.items():
    for ddi_id in ddi_ids:
        ddi_id_to_class[ddi_id] = class_id

def do_compute(batch, device, model, kg_features):
    '''
        *batch: (pos_tri, neg_tri, original_index)
        *pos/neg_tri: (batch_h, batch_t, batch_r)
    '''
    # 評価用（マルチクラス分類のAccuracy計算などに使用）
    preds_class, ground_truth_class = [], []
    
    pos_tri, neg_tri, original_index = batch
    
    # --- Positive ---
    pos_tri = [tensor.to(device=device) for tensor in pos_tri]
    # p_logits: [batch_size, 6]
    p_logits, attention_weights = model(pos_tri, kg_features)
    
    # batch_r をマルチクラス分類の正解ラベル (0~5) として取得
    p_targets = []
    for r in pos_tri[2][0].cpu().numpy():
        target_class = ddi_id_to_class.get(r.item(), 5) 
        p_targets.append(target_class)
    p_targets = torch.tensor(p_targets, dtype=torch.long).to(device)
    
    # 6クラス分類
    preds_class.append(torch.argmax(p_logits.detach(), dim=1).cpu().numpy())
    ground_truth_class.append(p_targets.cpu().numpy())

    # --- Negative ---
    neg_tri = [tensor.to(device=device) for tensor in neg_tri]
    # n_logits: [batch_size, 6]
    n_logits, attention_weights = model(neg_tri, kg_features)
    
    # 結合してNumPy配列化（Epoch終了時のAccuracyやF1スコア計算用）
    preds_class = np.concatenate(preds_class)
    ground_truth_class = np.concatenate(ground_truth_class)

    # Loss計算に必要なテンソルを返す
    return p_logits, p_targets, n_logits, preds_class, ground_truth_class


def do_compute_metrics(probas_pred, target):
    """
    probas_pred: [n_samples] (各クラス)
    target: [n_samples] (0~5 の正解ラベル)
    """
    
    # 2. Accuracy
    acc = metrics.accuracy_score(target, probas_pred)
    
    # 3. AUROC (One-vs-Rest で多クラス計算)

    
    # 4. F1, Precision, Recall (多クラスなので average='macro' を指定)
    f1_score = metrics.f1_score(target, probas_pred, average='macro')
    precision = metrics.precision_score(target, probas_pred, average='macro')
    recall = metrics.recall_score(target, probas_pred, average='macro')
    
    # PR曲線などの指標は多クラスでは少し複雑になるため、ここでは主要な4つを返します
    return acc, f1_score, precision, recall


# %% Model preparation
model = models.MVN_DDI(n_atom_feats, n_atom_hid, kge_dim, rel_total, heads_out_feat_params=[64,64,64,64], blocks_params=[2, 2, 2, 2], kg_emb_dim=200)
loss_fn = custom_loss.MultiClassSigmoidLoss()
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
    group="multi", 
    name="260303_proposed_multilabel",
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

        p_logits, p_targets, n_logits, preds_class, ground_truth_class = do_compute(batch, device, model, kg_features)


        train_probas_pred.append(preds_class)
        train_ground_truth.append(ground_truth_class)
        loss, loss_p, loss_n = loss_fn(p_logits, p_targets, n_logits)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(p_logits)
    train_loss /= len(train_data)

    with torch.no_grad():
        train_probas_pred = np.concatenate(train_probas_pred)
        train_ground_truth = np.concatenate(train_ground_truth)

        train_acc, train_f1, train_precision,train_recall = do_compute_metrics(train_probas_pred, train_ground_truth)

        # S1 scenario
        for batch in s1_data_loader:
            model.eval()
            p_logits, p_targets, n_logits, preds_class, ground_truth_class = do_compute(batch, device, model, kg_features)
            s1_probas_pred.append(preds_class)
            s1_ground_truth.append(ground_truth_class)
            loss, loss_p, loss_n = loss_fn(p_logits, p_targets, n_logits)
            s1_loss += loss.item() * len(p_logits)            

        s1_loss /= len(s1_data)
        s1_probas_pred = np.concatenate(s1_probas_pred)
        s1_ground_truth = np.concatenate(s1_ground_truth)
        s1_acc, s1_f1,s1_precision,s1_recall = do_compute_metrics(s1_probas_pred, s1_ground_truth)
        
        # S2 scenario
        for batch in s2_data_loader:
            model.eval()
            p_logits, p_targets, n_logits, preds_class, ground_truth_class = do_compute(batch, device, model, kg_features)
            s2_probas_pred.append(preds_class)
            s2_ground_truth.append(ground_truth_class)
            loss, loss_p, loss_n = loss_fn(p_logits, p_targets, n_logits)
            s2_loss += loss.item() * len(p_logits)            

        s2_loss /= len(s2_data)
        s2_probas_pred = np.concatenate(s2_probas_pred)
        s2_ground_truth = np.concatenate(s2_ground_truth)
        s2_acc, s2_f1,s2_precision,s2_recall = do_compute_metrics(s2_probas_pred, s2_ground_truth)

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
        'train_f1': train_f1,
        'train_precision': train_precision,
        'train_recall': train_recall,
        's1_acc': s1_acc,
        's1_f1': s1_f1,
        's1_precision': s1_precision,
        's1_recall': s1_recall,
        's2_acc': s2_acc,
        's2_f1': s2_f1,
        's2_precision': s2_precision,
        's2_recall': s2_recall,
    }
    logger(epoch=epoch, **loss_dict)
    
    print(f'Epoch: {epoch} ({time.time() - start:.4f}s), train_loss: {train_loss:.4f}, s1_loss: {s1_loss:.4f},s2_loss: {s2_loss:.4f}')
    print(f'\t\ttrain_acc: {train_acc:.4f}, train_f1: {train_f1:.4f},train_precision: {train_precision:.4f},train_recall:{train_recall:.4f}')
    print(f'\t\ts1_acc: {s1_acc:.4f}, s1_f1: {s1_f1:.4f}, s1_precision: {s1_precision:.4f}, s1_recall: {s1_recall:.4f}')
    print(f'\t\ts2_acc: {s2_acc:.4f}, s2_f1: {s2_f1:.4f}, s2_precision: {s2_precision:.4f}, s2_recall: {s2_recall:.4f}')

    # clear cache
    torch.cuda.empty_cache()


# %%
