# -*- coding: utf-8 -*-
"""
Created on 2026-03-02 (Mon) 16:46:11

DSN-DDI (2023) のベースラインスコアを算出する
モデルの学習は260227_dsnddi_abci.pyにて実行済み

チェックポイントのロードとtestの性能評価を行う

@author: I.Azuma
"""
# %%
BASE_DIR = "/home/aah18044co/github"
import os
import sys

os.chdir(f'{BASE_DIR}/XAI-DDI')
sys.path.append(f"{BASE_DIR}/XAI-DDI")

from baseline.DSN_DDI import models, custom_loss
from baseline.DSN_DDI.data_preprocessing import DrugDataset, DrugDataLoader

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

def do_compute(batch, device, model):
    """
    batch: (pos_tri, neg_tri, original_index)
    pos/neg_tri: (batch_h, batch_t, batch_r)
    """
    probas_pred, ground_truth = [], []
    pos_tri, neg_tri, original_index = batch

    # positive
    pos_tri = [tensor.to(device=device) for tensor in pos_tri]
    p_score = model(pos_tri)
    probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
    ground_truth.append(np.ones(len(p_score)))

    # negative
    neg_tri = [tensor.to(device=device) for tensor in neg_tri]
    n_score = model(neg_tri)
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

# %%
parser = argparse.ArgumentParser()

# paths / env
parser.add_argument("--base_dir", type=str, default="/home/aah18044co/github")
parser.add_argument("--fold_dir", type=str, default="/home/aah18044co/github/XAI-DDI/dataset/inductive_data/fold3")

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
sys.path.append(f"{BASE_DIR}/XAI-DDI")


from baseline.DSN_DDI import models, custom_loss
from baseline.DSN_DDI.data_preprocessing import DrugDataset, DrugDataLoader

# --- run naming / out dir ---
jobid = os.environ.get("PBS_JOBID", "nojobid")
ts = datetime.now().strftime("%y%m%d_%H%M%S")
run_name = args.run_name or f"dsnddi_{ts}_{jobid}"
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

# --- model ---
model = models.MVN_DDI(
    args.n_atom_feats, args.n_atom_hid, args.kge_dim, args.rel_total,
    heads_out_feat_params=[64, 64, 64, 64],
    blocks_params=[2, 2, 2, 2],
)
loss_fn = custom_loss.SigmoidLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** epoch)

model.to(device=device)

path =  f'{BASE_DIR}/XAI-DDI/workspace/abci_benchmarking/baselines/DSN-DDI/results/260227/fold3/dsnddi_260227_150115_1655383.pbs1/best_s1.pt'
ckpt = torch.load(path, map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])

s1_probas_pred = []
s1_ground_truth = []
s2_probas_pred = []
s2_ground_truth = []    

with torch.no_grad():
    for batch in s1_loader:
        _, _, probas_pred, ground_truth = do_compute(batch, device, model)
        s1_probas_pred.append(probas_pred)
        s1_ground_truth.append(ground_truth)
    s1_probas_pred = np.concatenate(s1_probas_pred)
    s1_ground_truth = np.concatenate(s1_ground_truth)
    s1_acc, s1_auc_roc, s1_f1, s1_precision, s1_recall, s1_int_ap, s1_ap = do_compute_metrics(s1_probas_pred, s1_ground_truth)

    for batch in s2_loader:
        _, _, probas_pred, ground_truth = do_compute(batch, device, model)
        s2_probas_pred.append(probas_pred)
        s2_ground_truth.append(ground_truth)
    s2_probas_pred = np.concatenate(s2_probas_pred)
    s2_ground_truth = np.concatenate(s2_ground_truth)
    s2_acc, s2_auc_roc, s2_f1, s2_precision, s2_recall, s2_int_ap, s2_ap = do_compute_metrics(s2_probas_pred, s2_ground_truth)  

print("=== DSN-DDI Baseline Evaluation ===")
print(f"S1: acc={s1_acc:.4f}, auc_roc={s1_auc_roc:.4f}, ap={s1_ap:.4f}, f1={s1_f1:.4f}, precision={s1_precision:.4f}, recall={s1_recall:.4f}, int_ap={s1_int_ap:.4f}")
print(f"S2: acc={s2_acc:.4f}, auc_roc={s2_auc_roc:.4f}, ap={s2_ap:.4f}, f1={s2_f1:.4f}, precision={s2_precision:.4f}, recall={s2_recall:.4f}, int_ap={s2_int_ap:.4f}")

"""
Results
fold0
S1: acc=0.6600, auc_roc=0.7347, ap=0.7350, f1=0.5902, precision=0.7427, recall=0.4896, int_ap=0.7349
S2: acc=0.7564, auc_roc=0.8383, ap=0.8369, f1=0.7370, precision=0.8008, recall=0.6827, int_ap=0.8369

fold1
S1: acc=0.6994, auc_roc=0.7762, ap=0.7754, f1=0.6561, precision=0.7666, recall=0.5734, int_ap=0.7754
S2: acc=0.7554, auc_roc=0.8356, ap=0.8308, f1=0.7426, precision=0.7837, recall=0.7057, int_ap=0.8308

fold2
S1: acc=0.6785, auc_roc=0.7513, ap=0.7551, f1=0.6244, precision=0.7508, recall=0.5344, int_ap=0.7551
S2: acc=0.7728, auc_roc=0.8524, ap=0.8516, f1=0.7587, precision=0.8089, recall=0.7144, int_ap=0.8516

fold3
S1: acc=0.7212, auc_roc=0.7926, ap=0.7955, f1=0.6866, precision=0.7839, recall=0.6108, int_ap=0.7954
S2: acc=0.7857, auc_roc=0.8602, ap=0.8566, f1=0.7875, precision=0.7809, recall=0.7942, int_ap=0.8566

"""

# %% time-split evaluation
parser = argparse.ArgumentParser()

# paths / env
parser.add_argument("--base_dir", type=str, default="/home/aah18044co/github")
parser.add_argument("--fold_dir", type=str, default="/home/aah18044co/github/XAI-DDI/dataset/drugbank/time_split")

# output
parser.add_argument("--out_dir", type=str, default="/home/aah18044co/github/XAI-DDI/workspace/abci_benchmarking/baselines/DSN-DDI/results/260302_timesplit/route2_260302_220835_1664008.pbs1")
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
sys.path.append(f"{BASE_DIR}/XAI-DDI")


from baseline.DSN_DDI import models, custom_loss
from baseline.DSN_DDI.data_preprocessing import DrugDataset, DrugDataLoader

# --- run naming / out dir ---
jobid = os.environ.get("PBS_JOBID", "nojobid")
ts = datetime.now().strftime("%y%m%d_%H%M%S")
run_name = args.run_name or f"dsnddi_{ts}_{jobid}"
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
df_ddi_s1 = pd.read_csv(os.path.join(fold_dir, "val.csv"))
df_ddi_s2 = pd.read_csv(os.path.join(fold_dir, "test.csv"))

s1_tup = [(h, t, r) for h, t, r in zip(df_ddi_s1["d1"], df_ddi_s1["d2"], df_ddi_s1["type"])]
s2_tup = [(h, t, r) for h, t, r in zip(df_ddi_s2["d1"], df_ddi_s2["d2"], df_ddi_s2["type"])]

s1_data = DrugDataset(s1_tup, disjoint_split=True)
s2_data = DrugDataset(s2_tup, disjoint_split=True)


s1_loader = DrugDataLoader(s1_data, batch_size=args.batch_size * 3, num_workers=0)
s2_loader = DrugDataLoader(s2_data, batch_size=args.batch_size * 3, num_workers=0)

# --- model ---
model = models.MVN_DDI(
    args.n_atom_feats, args.n_atom_hid, args.kge_dim, args.rel_total,
    heads_out_feat_params=[64, 64, 64, 64],
    blocks_params=[2, 2, 2, 2],
)
loss_fn = custom_loss.SigmoidLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** epoch)

model.to(device=device)

path =  f'{BASE_DIR}/XAI-DDI/workspace/abci_benchmarking/baselines/DSN-DDI/results/260302_timesplit/route2_260303_134318_1665257.pbs1/best_s1.pt'
ckpt = torch.load(path, map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])

s1_probas_pred = []
s1_ground_truth = []
s2_probas_pred = []
s2_ground_truth = []    

with torch.no_grad():
    for batch in s1_loader:
        _, _, probas_pred, ground_truth = do_compute(batch, device, model)
        s1_probas_pred.append(probas_pred)
        s1_ground_truth.append(ground_truth)
    s1_probas_pred = np.concatenate(s1_probas_pred)
    s1_ground_truth = np.concatenate(s1_ground_truth)
    s1_acc, s1_auc_roc, s1_f1, s1_precision, s1_recall, s1_int_ap, s1_ap = do_compute_metrics(s1_probas_pred, s1_ground_truth)

    for batch in s2_loader:
        _, _, probas_pred, ground_truth = do_compute(batch, device, model)
        s2_probas_pred.append(probas_pred)
        s2_ground_truth.append(ground_truth)
    s2_probas_pred = np.concatenate(s2_probas_pred)
    s2_ground_truth = np.concatenate(s2_ground_truth)
    s2_acc, s2_auc_roc, s2_f1, s2_precision, s2_recall, s2_int_ap, s2_ap = do_compute_metrics(s2_probas_pred, s2_ground_truth)  

print("=== DSN-DDI Baseline Evaluation ===")
print(f"S1: acc={s1_acc:.4f}, auc_roc={s1_auc_roc:.4f}, ap={s1_ap:.4f}, f1={s1_f1:.4f}, precision={s1_precision:.4f}, recall={s1_recall:.4f}, int_ap={s1_int_ap:.4f}")
print(f"S2: acc={s2_acc:.4f}, auc_roc={s2_auc_roc:.4f}, ap={s2_ap:.4f}, f1={s2_f1:.4f}, precision={s2_precision:.4f}, recall={s2_recall:.4f}, int_ap={s2_int_ap:.4f}")


# %%
