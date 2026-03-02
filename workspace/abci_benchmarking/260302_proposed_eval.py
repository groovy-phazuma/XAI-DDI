# -*- coding: utf-8 -*-
"""
Created on 2026-03-02 (Mon) 17:54:02

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

# %% functions
SEED = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def do_compute(batch, device, model, kg_features):
    """
    batch: (pos_tri, neg_tri, original_index)
    pos/neg_tri: (batch_h, batch_t, batch_r)
    """
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
    ap = metrics.average_precision_score(target, probas_pred)
    return acc, auroc, f1_score, precision, recall, int_ap, ap

def save_ckpt(path, epoch, model, optimizer, scheduler, best_s1_acc, args):
    obj = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_s1_acc": best_s1_acc,
        "args": vars(args),
    }
    torch.save(obj, path)

def load_ckpt(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_s1_acc = ckpt.get("best_s1_acc", 0.0)
    return start_epoch, best_s1_acc

# %% Test
parser = argparse.ArgumentParser()

# paths / env
parser.add_argument("--base_dir", type=str, default="/home/aah18044co/github")
parser.add_argument("--fold_dir", type=str, default="/home/aah18044co/github/XAI-DDI/dataset/inductive_data/fold3")
parser.add_argument("--kg_emb_path", type=str, default="/home/aah18044co/github/XAI-DDI/dataset/kg_embeddings/selected_genes_14662_embeddings.pkl")

# output
parser.add_argument("--out_dir", type=str, default="/home/aah18044co/github/XAI-DDI/workspace/abci_benchmarking/results/260223/fold3")
parser.add_argument("--run_name", type=str, default=None, help="if None, auto-generate by time/jobid")

# model hyperparams
parser.add_argument("--n_atom_feats", type=int, default=55)
parser.add_argument("--n_atom_hid", type=int, default=128)
parser.add_argument("--rel_total", type=int, default=86)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--kge_dim", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=128)

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


from route2 import models, custom_loss
from route2.data_preprocessing import DrugDataset, DrugDataLoader

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
df_ddi_s1 = pd.read_csv(os.path.join(fold_dir, "s1.csv"))
df_ddi_s2 = pd.read_csv(os.path.join(fold_dir, "s2.csv"))

s1_tup = [(h, t, r) for h, t, r in zip(df_ddi_s1["d1"], df_ddi_s1["d2"], df_ddi_s1["type"])]
s2_tup = [(h, t, r) for h, t, r in zip(df_ddi_s2["d1"], df_ddi_s2["d2"], df_ddi_s2["type"])]

s1_data = DrugDataset(s1_tup, disjoint_split=True)
s2_data = DrugDataset(s2_tup, disjoint_split=True)

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

path =  f'{BASE_DIR}/XAI-DDI/workspace/abci_benchmarking/results/260223/fold3/route2_260224_114841_1642383.pbs1/best_s1.pt'
ckpt = torch.load(path, map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])

s1_probas_pred = []
s1_ground_truth = []
s2_probas_pred = []
s2_ground_truth = []    

with torch.no_grad():
    for batch in s1_loader:
        p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model, kg_features)
        s1_probas_pred.append(probas_pred)
        s1_ground_truth.append(ground_truth)
    s1_probas_pred = np.concatenate(s1_probas_pred)
    s1_ground_truth = np.concatenate(s1_ground_truth)
    s1_acc, s1_auc_roc, s1_f1, s1_precision, s1_recall, s1_int_ap, s1_ap = do_compute_metrics(s1_probas_pred, s1_ground_truth)

    for batch in s2_loader:
        p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model, kg_features)
        s2_probas_pred.append(probas_pred)
        s2_ground_truth.append(ground_truth)
    s2_probas_pred = np.concatenate(s2_probas_pred)
    s2_ground_truth = np.concatenate(s2_ground_truth)
    s2_acc, s2_auc_roc, s2_f1, s2_precision, s2_recall, s2_int_ap, s2_ap = do_compute_metrics(s2_probas_pred, s2_ground_truth)  

print("=== DSN-DDI Baseline Evaluation ===")
print(f"S1: acc={s1_acc:.4f}, auc_roc={s1_auc_roc:.4f}, ap={s1_ap:.4f}, f1={s1_f1:.4f}, precision={s1_precision:.4f}, recall={s1_recall:.4f}, int_ap={s1_int_ap:.4f}")
print(f"S2: acc={s2_acc:.4f}, auc_roc={s2_auc_roc:.4f}, ap={s2_ap:.4f}, f1={s2_f1:.4f}, precision={s2_precision:.4f}, recall={s2_recall:.4f}, int_ap={s2_int_ap:.4f}")

# %%
"""
Results
fold0
***

fold1
S1: acc=0.7096, auc_roc=0.7888, ap=0.7789, f1=0.6914, precision=0.7377, recall=0.6505, int_ap=0.7789
S2: acc=0.7724, auc_roc=0.8539, ap=0.8431, f1=0.7780, precision=0.7593, recall=0.7976, int_ap=0.8431

fold2
S1: acc=0.6642, auc_roc=0.7348, ap=0.7291, f1=0.6155, precision=0.7198, recall=0.5376, int_ap=0.7290
S2: acc=0.7513, auc_roc=0.8307, ap=0.8210, f1=0.7398, precision=0.7756, recall=0.7072, int_ap=0.8210

fold3
S1: acc=0.7316, auc_roc=0.8084, ap=0.8043, f1=0.7152, precision=0.7619, recall=0.6738, int_ap=0.8043
S2: acc=0.7620, auc_roc=0.8398, ap=0.8296, f1=0.7696, precision=0.7459, recall=0.7948, int_ap=0.8296

"""