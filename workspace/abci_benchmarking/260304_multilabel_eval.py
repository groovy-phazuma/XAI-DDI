# -*- coding: utf-8 -*-
"""
Created on 2026-03-04 (Wed) 15:03:18

multilabel classfication taskにおいて、kg featuresのattention-scoreを回収する

@author: I.Azuma
"""
# %%
import os
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

# =========================
# Utils
# =========================
SEED = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

ddi_id_to_class = {}
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
    p_logits, p_attn_weight = model(pos_tri, kg_features)
    
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
    n_logits, n_attn_weight = model(neg_tri, kg_features)
    
    # 結合してNumPy配列化（Epoch終了時のAccuracyやF1スコア計算用）
    preds_class = np.concatenate(preds_class)
    ground_truth_class = np.concatenate(ground_truth_class)

    # Loss計算に必要なテンソルを返す
    return p_logits, p_targets, n_logits, preds_class, ground_truth_class, p_attn_weight, n_attn_weight


def do_compute_metrics(probas_pred, target):
    """
    probas_pred: [n_samples] (各クラス)
    target: [n_samples] (0~5 の正解ラベル)
    """
    
    # 2. Accuracy
    acc = metrics.accuracy_score(target, probas_pred)
    
    # 4. F1, Precision, Recall (多クラスなので average='macro' を指定)
    f1_score = metrics.f1_score(target, probas_pred, average='macro')
    precision = metrics.precision_score(target, probas_pred, average='macro')
    recall = metrics.recall_score(target, probas_pred, average='macro')

    return acc, f1_score, precision, recall

# %%
parser = argparse.ArgumentParser()

# paths / env
parser.add_argument("--base_dir", type=str, default="/home/aah18044co/github")
parser.add_argument("--fold_dir", type=str, default="/home/aah18044co/github/XAI-DDI/dataset/inductive_data/fold2")
parser.add_argument("--kg_emb_path", type=str, default="/home/aah18044co/github/XAI-DDI/dataset/kg_embeddings/selected_genes_14662_embeddings.pkl")

# output
parser.add_argument("--out_dir", type=str, default="/home/aah18044co/github/XAI-DDI/workspace/abci_benchmarking/results/260303_multi/fold2")
parser.add_argument("--run_name", type=str, default=None, help="if None, auto-generate by time/jobid")

# model hyperparams
parser.add_argument("--n_atom_feats", type=int, default=55)
parser.add_argument("--n_atom_hid", type=int, default=128)
parser.add_argument("--rel_total", type=int, default=86)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--kge_dim", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=1024)

parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--neg_samples", type=int, default=1)
parser.add_argument("--data_size_ratio", type=int, default=1)
parser.add_argument("--use_cuda", type=int, default=1, choices=[0, 1])

# resume
parser.add_argument("--resume_ckpt", type=str, default=None, help="checkpoint .pt to resume")

args = parser.parse_args([])

# --- seed / device ---
set_seed(SEED)
device = "cuda:0" if (torch.cuda.is_available() and args.use_cuda == 1) else "cpu"

# --- paths ---
BASE_DIR = args.base_dir
import os
os.chdir(f'{BASE_DIR}/XAI-DDI')
sys.path.append(f"{BASE_DIR}/XAI-DDI/model")


from multi import models, custom_loss
from multi.data_preprocessing import DrugDataset, DrugDataLoader

# --- run naming / out dir ---
jobid = os.environ.get("PBS_JOBID", "nojobid")
ts = datetime.now().strftime("%y%m%d_%H%M%S")
run_name = args.run_name or f"route2_{ts}_{jobid}"
out_dir = os.path.join(BASE_DIR, args.out_dir)
run_dir = os.path.join(out_dir, run_name)
os.makedirs(run_dir, exist_ok=True)

# files
best_ckpt_path = os.path.join(run_dir, "best_s1.pt")
last_ckpt_path = os.path.join(run_dir, "last.pt")
log_path = os.path.join(run_dir, "stdout.log")

print("===== RUN INFO =====")
print("run_name:", run_name)
print("run_dir :", run_dir)
print("device  :", device)
print("args    :", args)

# --- dataset ---
fold_dir = os.path.join(BASE_DIR, args.fold_dir)
df_ddi_train = pd.read_csv(os.path.join(fold_dir, "train.csv"))
df_ddi_s1 = pd.read_csv(os.path.join(fold_dir, "s1.csv"))
df_ddi_s2 = pd.read_csv(os.path.join(fold_dir, "s2.csv"))

train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train["d1"], df_ddi_train["d2"], df_ddi_train["type"])]
s1_tup = [(h, t, r) for h, t, r in zip(df_ddi_s1["d1"], df_ddi_s1["d2"], df_ddi_s1["type"])]
s2_tup = [(h, t, r) for h, t, r in zip(df_ddi_s2["d1"], df_ddi_s2["d2"], df_ddi_s2["type"])]

train_data = DrugDataset(train_tup, ratio=args.data_size_ratio, neg_ent=args.neg_samples)
s1_data = DrugDataset(s1_tup, disjoint_split=True)
s2_data = DrugDataset(s2_tup, disjoint_split=True)

print(f"Training with {len(train_data)} samples, s1 with {len(s1_data)}, and s2 with {len(s2_data)}")

train_loader = DrugDataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
s1_loader = DrugDataLoader(s1_data, batch_size=args.batch_size * 3, num_workers=0)
s2_loader = DrugDataLoader(s2_data, batch_size=args.batch_size * 3, num_workers=0)

# KG embeddings
kg_path = os.path.join(BASE_DIR, args.kg_emb_path)
kg_features = pd.read_pickle(kg_path)

# --- model ---
model = models.MVN_DDI(
    args.n_atom_feats, args.n_atom_hid, args.kge_dim, args.rel_total,
    heads_out_feat_params=[64, 64, 64, 64],
    blocks_params=[2, 2, 2, 2],
    kg_emb_dim=200
)

model.to(device=device)

# load checkpoint
path = '/home/aah18044co/github/XAI-DDI/workspace/abci_benchmarking/results/260303_multi/fold1/route2_260303_205956_1667589.pbs1/best_s1.pt'
path = '/home/aah18044co/github/XAI-DDI/workspace/abci_benchmarking/results/260303_multi/fold2/route2_260303_224520_1667835.pbs1/best_s1.pt'
chk_pt = torch.load(path, map_location=device)
model.load_state_dict(chk_pt['model_state_dict'])


# %% test
s1_probas_pred = []
s1_ground_truth = []
s2_probas_pred = []
s2_ground_truth = []    

overall_p_attn = []
overall_n_attn = []
with torch.no_grad():
    """
    for batch in s1_loader:
        p_logits, p_targets, n_logits, preds_class, ground_truth_class, _, _ = do_compute(batch, device, model, kg_features)
        s1_probas_pred.append(preds_class)
        s1_ground_truth.append(ground_truth_class)
    s1_probas_pred = np.concatenate(s1_probas_pred)
    s1_ground_truth = np.concatenate(s1_ground_truth)
    s1_acc, s1_f1, s1_precision, s1_recall = do_compute_metrics(s1_probas_pred, s1_ground_truth)
    """

    for batch in s2_loader:
        p_logits, p_targets, n_logits, preds_class, ground_truth_class, p_attn_weight, n_attn_weight = do_compute(batch, device, model, kg_features)
        s2_probas_pred.append(preds_class)
        s2_ground_truth.append(ground_truth_class)

        p_attn_weight = p_attn_weight[:,0,:].cpu().numpy()
        n_attn_weight = n_attn_weight[:,0,:].cpu().numpy()
        overall_p_attn.append(p_attn_weight)
        overall_n_attn.append(n_attn_weight)
    s2_probas_pred = np.concatenate(s2_probas_pred)
    s2_ground_truth = np.concatenate(s2_ground_truth)
    s2_acc, s2_f1, s2_precision, s2_recall = do_compute_metrics(s2_probas_pred, s2_ground_truth)

print("=== Evaluation ===")
#print(f"S1: acc={s1_acc:.4f}, ap={s1_precision:.4f}, f1={s1_f1:.4f}, recall={s1_recall:.4f}")
print(f"S2: acc={s2_acc:.4f}, ap={s2_precision:.4f}, f1={s2_f1:.4f}, recall={s2_recall:.4f}")

pd.to_pickle(overall_p_attn, os.path.join(run_dir, "overall_p_attn.pkl"))
pd.to_pickle(overall_n_attn, os.path.join(run_dir, "overall_n_attn.pkl"))

# %%
"""
last batch in s2_loader
>> shuffle=False

(DataBatch(x=[31199, 128], edge_index=[2, 66944], batch=[31199], ptr=[1195]),
 DataBatch(x=[31859, 128], edge_index=[2, 68520], batch=[31859], ptr=[1195]),
 tensor([[48, 46, 15,  ..., 72, 48, 48]]),
 BipartiteDataBatch(edge_index=[2, 858983], x_s=[31199, 55], x_t=[31859, 55], batch=[38811], ptr=[1195]))
"""
