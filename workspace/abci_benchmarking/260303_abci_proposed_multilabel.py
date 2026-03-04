# -*- coding: utf-8 -*-
"""
Created on 2026-02-09
Route2 training script for ABCI batch job

batch_size=1024

@author: I.Azuma
"""
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

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    # paths / env
    parser.add_argument("--base_dir", type=str, default="/home/aah18044co/github")
    parser.add_argument("--fold_dir", type=str, default="/home/aah18044co/github/XAI-DDI/dataset/inductive_data/fold1")
    parser.add_argument("--kg_emb_path", type=str, default="/home/aah18044co/github/XAI-DDI/dataset/kg_embeddings/selected_genes_14662_embeddings.pkl")

    # output
    parser.add_argument("--out_dir", type=str, default="/home/aah18044co/github/XAI-DDI/workspace/abci_benchmarking/results/260223/fold1")
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

    args = parser.parse_args()

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
    loss_fn = custom_loss.MultiClassSigmoidLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** epoch)

    model.to(device=device)

    # --- logger ---
    option_list = defaultdict(list)
    for k, v in vars(args).items():
        option_list[k] = v

    # --- resume if needed ---
    start_epoch = 1
    best_s1_acc = 0.0
    if args.resume_ckpt is not None and os.path.exists(args.resume_ckpt):
        print(f"[RESUME] loading: {args.resume_ckpt}")
        start_epoch, best_s1_acc = load_ckpt(
            args.resume_ckpt, model, optimizer=optimizer, scheduler=scheduler,
            map_location=device
        )
        print(f"[RESUME] start_epoch={start_epoch}, best_s1_acc={best_s1_acc:.4f}")

    # --- train ---
    print("Starting training at", datetime.today())
    for epoch in range(start_epoch, args.n_epochs + 1):
        start = time.time()
        train_loss = 0.0
        s1_loss = 0.0
        s2_loss = 0.0

        train_probas_pred, train_ground_truth = [], []
        s1_probas_pred, s1_ground_truth = [], []
        s2_probas_pred, s2_ground_truth = [], []

        # train loop
        for batch in train_loader:
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

        # eval
        with torch.no_grad():
            train_probas_pred = np.concatenate(train_probas_pred)
            train_ground_truth = np.concatenate(train_ground_truth)

            train_acc, train_f1, train_precision,train_recall = do_compute_metrics(train_probas_pred, train_ground_truth)

            # S1
            for batch in s1_loader:
                model.eval()
                s1_probas_pred.append(preds_class)
                s1_ground_truth.append(ground_truth_class)
                loss, _, _ = loss_fn(p_logits, p_targets, n_logits)
                s1_loss += loss.item() * len(p_logits)

            s1_loss /= len(s1_data)
            s1_probas_pred = np.concatenate(s1_probas_pred)
            s1_ground_truth = np.concatenate(s1_ground_truth)
            s1_acc, s1_f1, s1_precision, s1_recall = do_compute_metrics(s1_probas_pred, s1_ground_truth)

            # S2
            for batch in s2_loader:
                model.eval()
                p_logits, p_targets, n_logits, preds_class, ground_truth_class = do_compute(batch, device, model, kg_features)
                s2_probas_pred.append(preds_class)
                s2_ground_truth.append(ground_truth_class)
                loss, _, _ = loss_fn(p_logits, p_targets, n_logits)
                s2_loss += loss.item() * len(p_logits)

            s2_loss /= len(s2_data)
            s2_probas_pred = np.concatenate(s2_probas_pred)
            s2_ground_truth = np.concatenate(s2_ground_truth)
            s2_acc,s2_f1,s2_precision,s2_recall = do_compute_metrics(s2_probas_pred, s2_ground_truth)

        # scheduler
        if scheduler is not None:
            scheduler.step()

        # checkpoint: always save last
        save_ckpt(last_ckpt_path, epoch, model, optimizer, scheduler, best_s1_acc, args)

        # checkpoint: save best on s1_acc
        if s1_acc > best_s1_acc:
            best_s1_acc = s1_acc
            save_ckpt(best_ckpt_path, epoch, model, optimizer, scheduler, best_s1_acc, args)
            print(f"[BEST] epoch={epoch} best_s1_acc={best_s1_acc:.4f} saved: {best_ckpt_path}")

        # log
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

        print(f"Epoch: {epoch} ({time.time() - start:.2f}s), train_loss: {train_loss:.4f}, s1_loss: {s1_loss:.4f}, s2_loss: {s2_loss:.4f}")
        print(f"\ttrain_acc: {train_acc:.4f}, train_f1: {train_f1:.4f}, train_precision: {train_precision:.4f}, train_recall: {train_recall:.4f}")
        print(f"\ts1_acc: {s1_acc:.4f}, s1_f1: {s1_f1:.4f}, s1_precision: {s1_precision:.4f}, s1_recall: {s1_recall:.4f}")
        print(f"\ts2_acc: {s2_acc:.4f}, s2_f1: {s2_f1:.4f}, s2_precision: {s2_precision:.4f}, s2_recall: {s2_recall:.4f}")

        torch.cuda.empty_cache()

    print("\nTraining finished.")
    print("best checkpoint:", best_ckpt_path)
    print("last checkpoint:", last_ckpt_path)

if __name__ == "__main__":
    main()
