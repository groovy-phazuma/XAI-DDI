#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSN-DDI (MUFFIN/SKGDDI) transductive evaluation on ABCI
- train/val/test split (val is split from original train)
- save split indices/data
- save best checkpoint by val metric
- load best checkpoint and evaluate on test

@author: I.Azuma
"""

import os
import sys
import json
import random
import argparse
import numpy as np
from time import time
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.utils.data as Data


# -------------------------
# Args
# -------------------------
def build_args():
    parser = argparse.ArgumentParser(description="Run MUFFIN/SKGDDI on ABCI (transductive) with train/val/test")

    # paths
    parser.add_argument("--base_dir", type=str, default="/home/aah18044co")
    parser.add_argument("--data_dir", type=str, default="/home/aah18044co/github/MUFFIN/data")

    parser.add_argument("--data_name", type=str, default="DrugBank", choices=["DrugBank", "DRKG"])
    parser.add_argument("--kg_file", type=str, default="/home/aah18044co/github/XAI-DDI/dataset/muffin_data/train.tsv")
    parser.add_argument("--graph_embedding_file", type=str, default="/home/aah18044co/github/MUFFIN/data/DRKG/gin_supervised_masking_embedding.npy")
    parser.add_argument("--entity_embedding_file", type=str, default="/home/aah18044co/github/XAI-DDI/dataset/muffin_data/entity_embeddings.npy")
    parser.add_argument("--relation_embedding_file", type=str, default="/home/aah18044co/github/XAI-DDI/dataset/muffin_data/relation_embeddings.npy")

    # pretrain
    parser.add_argument("--use_pretrain", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--pretrain_model_path", type=str, default="trained_model/model.pth")

    # training
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--stopping_steps", type=int, default=10)

    # batch sizes
    parser.add_argument("--DDI_batch_size", type=int, default=4096)
    parser.add_argument('--kg_batch_size', type=int, default=4096)
    parser.add_argument("--DDI_evaluate_size", type=int, default=4096)

    # model dims
    parser.add_argument("--entity_dim", type=int, default=256)
    parser.add_argument("--relation_dim", type=int, default=256)

    parser.add_argument("--aggregation_type", type=str, default="sum", choices=["sum", "concat", "pna"])
    parser.add_argument("--conv_dim_list", type=str, default="[64, 32, 16]")
    parser.add_argument("--mess_dropout", type=str, default="[0.1, 0.1, 0.1]")

    parser.add_argument("--kg_l2loss_lambda", type=float, default=1e-5)
    parser.add_argument("--DDI_l2loss_lambda", type=float, default=1e-5)

    parser.add_argument("--ddi_print_every", type=int, default=50)
    parser.add_argument("--evaluate_every", type=int, default=10)

    parser.add_argument("--multi_type", type=str, default="False")
    parser.add_argument("--n_hidden_1", type=int, default=512)
    parser.add_argument("--n_hidden_2", type=int, default=512)
    parser.add_argument("--out_dim", type=int, default=1)
    parser.add_argument("--structure_dim", type=int, default=300)
    parser.add_argument("--pre_entity_dim", type=int, default=200)
    parser.add_argument("--feature_fusion", type=str, default="init_double", choices=["concat", "sum", "init_double"])

    # device / memory
    parser.add_argument("--gpu_mem_fraction", type=float, default=0.8)

    # folds
    parser.add_argument("--fold", type=int, default=0,
                        help="0 means run all folds (1..5). 1..5 means run only that fold (recommended with PBS array).")

    # split
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="val split ratio from original train set per fold")
    parser.add_argument("--split_seed", type=int, default=None,
                        help="seed for splitting. if None, use --seed")

    # mode
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="train: train + save best + evaluate test. test: load checkpoint and evaluate test only")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="used in --mode test. path to checkpoint (.pth)")

    # outputs
    parser.add_argument("--out_root", type=str, default="/home/aah18044co/workspace/Baseline/MUFFIN/results/abci_runs",
                        help="root directory for outputs on ABCI")
    parser.add_argument("--run_name", type=str, default=None, help="if None, auto by timestamp+PBS_JOBID")

    # metric
    parser.add_argument("--select_metric", type=str, default="auc", choices=["auc", "acc"],
                        help="metric for best checkpoint selection in binary classification (recommend auc).")

    return parser.parse_args()


def make_run_dir(args):
    jobid = os.environ.get("PBS_JOBID", "nojobid")
    ts_str = __import__("datetime").datetime.now().strftime("%y%m%d_%H%M%S")
    run_name = args.run_name or f"muffin_{args.data_name}_{ts_str}_{jobid}"
    run_dir = os.path.join(args.out_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# -------------------------
# Split helpers
# -------------------------
def split_train_val(X, Y, val_ratio, seed):
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_val = int(round(n * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    return train_idx, val_idx


def save_split_npz(path, X, Y):
    # X,Y are torch tensors or numpy arrays
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.cpu().numpy()
    np.savez_compressed(path, X=X, Y=Y)


# -------------------------
# Checkpoint helpers
# -------------------------
def save_checkpoint(path, model, optimizer, epoch, best_metric, args, extra=None):
    obj = {
        "epoch": epoch,
        "best_metric": float(best_metric),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "args": vars(args),
        "extra": extra or {},
    }
    torch.save(obj, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# -------------------------
# Main
# -------------------------
def main():
    args = build_args()
    split_seed = args.seed if args.split_seed is None else args.split_seed

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
        try:
            torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
        except Exception as e:
            print("[WARN] set_per_process_memory_fraction failed:", e)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # import MUFFIN baseline modules
    muffin_path = os.path.join(args.base_dir, "github", "XAI-DDI", "baseline", "MUFFIN")
    sys.path.insert(0, muffin_path)

    from muffin_model import GCNModel
    from dataset import DataLoaderSKGDDI
    from utils import evaluate, save_model, load_model, early_stopping

    # outputs
    run_dir = make_run_dir(args)
    print("run_dir:", run_dir)

    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    base_save_dir = os.path.join(run_dir, "checkpoints")
    splits_dir = os.path.join(run_dir, "splits")
    metrics_dir = os.path.join(run_dir, "metrics")
    os.makedirs(base_save_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # init data
    data = DataLoaderSKGDDI(args)
    n_approved_drug = data.n_approved_drug

    # pretrain embeddings
    if args.use_pretrain == 1 and args.feature_fusion in ["sum", "concat", "init_double"]:
        structure_pre_embed = torch.tensor(data.structure_pre_embed).to(device)
        entity_pre_embed = torch.tensor(data.entity_pre_embed).to(device).float()
        relation_pre_embed = torch.tensor(data.relation_pre_embed).to(device).float()
        embedding_pre = torch.LongTensor(range(data.n_approved_drug)).to(device)
        embedding_after = torch.LongTensor(range(data.n_approved_drug, data.n_entities)).to(device)
    else:
        entity_pre_embed, relation_pre_embed, structure_pre_embed = None, None, None
        embedding_pre, embedding_after = None, None

    train_graph = None

    # fold control
    if args.fold == 0:
        fold_indices = list(range(5))
    else:
        if not (1 <= args.fold <= 5):
            raise ValueError("--fold must be 0 or 1..5")
        fold_indices = [args.fold - 1]

    # -------------------------
    # MODE: test only
    # -------------------------
    if args.mode == "test":
        if args.ckpt_path is None:
            raise ValueError("--mode test requires --ckpt_path")

        # infer fold from args.fold (required)
        if args.fold == 0:
            raise ValueError("--mode test requires --fold 1..5 (single fold)")
        i = fold_indices[0]

        fold_save_dir = os.path.join(base_save_dir, f"fold{i+1}")
        os.makedirs(fold_save_dir, exist_ok=True)

        # build model
        model = GCNModel(args, data.n_entities, data.n_relations,
                        entity_pre_embed, relation_pre_embed, structure_pre_embed)
        model.to(device)
        load_checkpoint(args.ckpt_path, model, optimizer=None, map_location=device)

        # load test split (saved npz) if exists; else fall back to original test
        test_npz = os.path.join(splits_dir, f"fold{i+1}_test.npz")
        if os.path.exists(test_npz):
            z = np.load(test_npz)
            test_x = torch.from_numpy(z["X"])
            test_y = torch.from_numpy(z["Y"])
        else:
            test_x = torch.from_numpy(data.DDI_test_data_X[i])
            test_y = torch.from_numpy(data.DDI_test_data_Y[i])

        loader_test = Data.DataLoader(
            dataset=Data.TensorDataset(test_x, test_y),
            batch_size=args.DDI_evaluate_size,
            shuffle=False
        )
        loader_idx = Data.DataLoader(
            dataset=Data.TensorDataset(torch.LongTensor(range(n_approved_drug))),
            batch_size=16,
            shuffle=False
        )

        model.eval()
        with torch.no_grad():
            if args.multi_type == "False":
                precision, recall, f1, acc, auc, _ = evaluate(
                    args, model, train_graph, loader_test,
                    embedding_pre, embedding_after, loader_idx, epoch=0
                )
                result = {"precision": float(precision), "recall": float(recall), "f1": float(f1), "acc": float(acc), "auc": float(auc)}
            else:
                macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, acc, _ = evaluate(
                    args, model, train_graph, loader_test,
                    embedding_pre, embedding_after, loader_idx, epoch=0
                )
                result = {"macro_precision": float(macro_p), "macro_recall": float(macro_r), "macro_f1": float(macro_f1),
                          "micro_precision": float(micro_p), "micro_recall": float(micro_r), "micro_f1": float(micro_f1),
                          "acc": float(acc)}

        out_json = os.path.join(metrics_dir, f"fold{i+1}_test_from_ckpt.json")
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print("[TEST ONLY] result:", result)
        print("saved:", out_json)
        return

    # -------------------------
    # MODE: train (train/val/test)
    # -------------------------
    all_test_results = []

    for i in fold_indices:
        print("=" * 80)
        print(f"Fold {i+1}/5  (split_seed={split_seed}, val_ratio={args.val_ratio})")
        torch.cuda.empty_cache()

        fold_save_dir = os.path.join(base_save_dir, f"fold{i+1}")
        os.makedirs(fold_save_dir, exist_ok=True)

        # prepare base tensors from original fold split
        # NOTE: original code uses train_x/train_y and test_x/test_y from DataLoaderSKGDDI
        full_train_x = torch.from_numpy(data.DDI_train_data_X[i])
        full_train_y = torch.from_numpy(data.DDI_train_data_Y[i])
        test_x = torch.from_numpy(data.DDI_test_data_X[i])
        test_y = torch.from_numpy(data.DDI_test_data_Y[i])

        # split train->train/val
        train_idx, val_idx = split_train_val(full_train_x.numpy(), full_train_y.numpy(), args.val_ratio, seed=split_seed + i)
        train_x = full_train_x[train_idx]
        train_y = full_train_y[train_idx]
        val_x = full_train_x[val_idx]
        val_y = full_train_y[val_idx]

        # save split datasets
        save_split_npz(os.path.join(splits_dir, f"fold{i+1}_train.npz"), train_x, train_y)
        save_split_npz(os.path.join(splits_dir, f"fold{i+1}_val.npz"), val_x, val_y)
        save_split_npz(os.path.join(splits_dir, f"fold{i+1}_test.npz"), test_x, test_y)

        # dataloaders
        loader_train = Data.DataLoader(
            dataset=Data.TensorDataset(train_x, train_y),
            batch_size=args.DDI_batch_size,
            shuffle=True
        )
        loader_val = Data.DataLoader(
            dataset=Data.TensorDataset(val_x, val_y),
            batch_size=args.DDI_evaluate_size,
            shuffle=False
        )
        loader_test = Data.DataLoader(
            dataset=Data.TensorDataset(test_x, test_y),
            batch_size=args.DDI_evaluate_size,
            shuffle=False
        )
        loader_idx = Data.DataLoader(
            dataset=Data.TensorDataset(torch.LongTensor(range(n_approved_drug))),
            batch_size=16,
            shuffle=False
        )

        # model
        model = GCNModel(args, data.n_entities, data.n_relations, entity_pre_embed, relation_pre_embed, structure_pre_embed)
        if args.use_pretrain == 2:
            model = load_model(model, args.pretrain_model_path)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        if args.multi_type != "False":
            loss_func = torch.nn.CrossEntropyLoss()
        else:
            loss_func = torch.nn.BCEWithLogitsLoss()

        best_metric = -1e18
        best_ckpt_path = os.path.join(fold_save_dir, "best.pt")
        last_ckpt_path = os.path.join(fold_save_dir, "last.pt")

        # for early stopping: track val metric
        val_metric_list = []

        time0 = time()

        for epoch in tqdm(range(1, args.n_epoch + 1)):
            model.train()
            ddi_total_loss = 0.0

            # train epoch
            for step, (batch_x, batch_y) in enumerate(loader_train, start=1):
                if torch.cuda.is_available():
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                out = model("calc_ddi_loss", train_graph, batch_x, embedding_pre, embedding_after, loader_idx, epoch)
                if args.multi_type == "False":
                    out = out.squeeze(-1)
                    loss = loss_func(out, batch_y.float())
                else:
                    loss = loss_func(out, batch_y.long())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ddi_total_loss += loss.item()

                if (step % args.ddi_print_every) == 0:
                    print(f"[TRAIN] epoch {epoch:04d} step {step:04d} loss {loss.item():.4f} mean {ddi_total_loss/step:.4f}")

            # save last checkpoint each epoch (cheap, safe)
            save_checkpoint(last_ckpt_path, model, optimizer, epoch, best_metric, args, extra={"fold": i+1})

            print(f"[TRAIN] epoch {epoch:04d} mean_loss {ddi_total_loss/max(1, len(loader_train)):.4f} elapsed {time()-time0:.1f}s")

            # evaluate on VAL
            if (epoch % args.evaluate_every) == 0:
                torch.cuda.empty_cache()
                model.eval()
                with torch.no_grad():
                    if args.multi_type == "False":
                        p, r, f1, acc, auc, all_embed = evaluate(
                            args, model, train_graph, loader_val,
                            embedding_pre, embedding_after, loader_idx, epoch
                        )
                        metric = auc if args.select_metric == "auc" else acc
                        print(f"[VAL] epoch {epoch:04d} P {p:.4f} R {r:.4f} F1 {f1:.4f} ACC {acc:.4f} AUC {auc:.4f} | select={args.select_metric}:{metric:.4f}")
                    else:
                        macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, acc, all_embed = evaluate(
                            args, model, train_graph, loader_val,
                            embedding_pre, embedding_after, loader_idx, epoch
                        )
                        metric = acc
                        print(f"[VAL] epoch {epoch:04d} MacroF1 {macro_f1:.4f} MicroF1 {micro_f1:.4f} ACC {acc:.4f}")

                val_metric_list.append(float(metric))
                best_now, should_stop = early_stopping(val_metric_list, args.stopping_steps)

                # save best by val metric
                if val_metric_list.index(best_now) == len(val_metric_list) - 1:
                    best_metric = float(metric)
                    # NOTE: if you want to keep using save_model() behavior, you can call it too.
                    # args.save_dir = fold_save_dir + "/"
                    # save_model(all_embed, model, args.save_dir, epoch, -1)
                    save_checkpoint(best_ckpt_path, model, optimizer, epoch, best_metric, args, extra={"fold": i+1})
                    print(f"[BEST] saved best checkpoint: {best_ckpt_path} (metric={best_metric:.4f})")

                if should_stop:
                    print(f"[EARLY STOP] fold {i+1} best_metric={best_now:.4f}")
                    break

        # -------- After training: load BEST and evaluate TEST --------
        print(f"[LOAD BEST] {best_ckpt_path}")
        load_checkpoint(best_ckpt_path, model, optimizer=None, map_location=device)

        model.eval()
        with torch.no_grad():
            if args.multi_type == "False":
                p, r, f1, acc, auc, _ = evaluate(
                    args, model, train_graph, loader_test,
                    embedding_pre, embedding_after, loader_idx, epoch=0
                )
                test_result = {"fold": i+1, "precision": float(p), "recall": float(r), "f1": float(f1), "acc": float(acc), "auc": float(auc)}
                print(f"[TEST] fold {i+1} P {p:.4f} R {r:.4f} F1 {f1:.4f} ACC {acc:.4f} AUC {auc:.4f}")
            else:
                macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, acc, _ = evaluate(
                    args, model, train_graph, loader_test,
                    embedding_pre, embedding_after, loader_idx, epoch=0
                )
                test_result = {"fold": i+1,
                               "macro_precision": float(macro_p), "macro_recall": float(macro_r), "macro_f1": float(macro_f1),
                               "micro_precision": float(micro_p), "micro_recall": float(micro_r), "micro_f1": float(micro_f1),
                               "acc": float(acc)}
                print(f"[TEST] fold {i+1} MacroF1 {macro_f1:.4f} MicroF1 {micro_f1:.4f} ACC {acc:.4f}")

        all_test_results.append(test_result)
        with open(os.path.join(metrics_dir, f"fold{i+1}_test.json"), "w") as f:
            json.dump(test_result, f, indent=2)

        # cleanup
        del model
        torch.cuda.empty_cache()

    # summary
    out_json = os.path.join(metrics_dir, "test_summary.json")
    with open(out_json, "w") as f:
        json.dump(all_test_results, f, indent=2)
    print("[DONE] saved summary:", out_json)
    print("Outputs saved under:", run_dir)


if __name__ == "__main__":
    main()