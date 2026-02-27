#!/usr/bin/env python3
"""
Created on 2025-08-05 (Tue) 11:12:36

DSN-DDIのtransductiveな問題設定における性能評価

データの準備: 250802_benchmark_preparation.py
KGの埋め込み: 250805_mykg_embedding.py

localのPCではメモリの問題で実行できなかったため、ABCIのGPU環境で実行
- batch_size: 128
    - 128で回ることを確認した
    - 1024でも大丈夫そう

- lr: 0.001
- epoch: 100

@author: I.Azuma
"""
# %%
BASE_DIR = '/home/aah18044co'
DATA_DIR = f'{BASE_DIR}/github/MUFFIN/data'

import random
import numpy as np

from time import time
from tqdm import tqdm

import torch.optim as optim
import torch.utils.data as Data

import argparse


# print gpu device name
import torch
print("Using GPU: ", torch.cuda.get_device_name(0))

# Clear GPU cache before starting
torch.cuda.empty_cache()

# Set memory fraction to avoid fragmentation
torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory

# Enable memory efficient settings
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Import AMP for mixed precision training
from torch.cuda.amp import autocast, GradScaler

import sys
sys.path.append(f'{BASE_DIR}/github/XAI-DDI/baseline/MUFFIN')
from muffin_model import GCNModel
from dataset import DataLoaderSKGDDI
from utils import evaluate, save_model, load_model, early_stopping

# %%
def parse_SKGDDI_args():
    parser = argparse.ArgumentParser(description="Run SKGDDI.")

    parser.add_argument('--seed', type=int, default=2020,
                        help='Random seed.')

    parser.add_argument('--data_name', nargs='?', default='DrugBank',help='Choose a dataset from {DrugBank, DRKG}')
    parser.add_argument('--data_dir', nargs='?', default=DATA_DIR+'/', help='Input data path.')
    parser.add_argument('--kg_file', nargs='?', default=f'{BASE_DIR}/github/XAI-DDI/dataset/muffin_data/train.tsv', help='KG file path.')
    parser.add_argument('--graph_embedding_file', nargs='?', default=f'{DATA_DIR}/DRKG/gin_supervised_masking_embedding.npy')
    parser.add_argument('--entity_embedding_file', nargs='?', default=f'{BASE_DIR}/github/XAI-DDI/dataset/muffin_data/entity_embeddings.npy')
    parser.add_argument('--relation_embedding_file', nargs='?', default=f'{BASE_DIR}/github/XAI-DDI/dataset/muffin_data/relation_embeddings.npy')

    parser.add_argument('--use_pretrain', type=int, default=1, help='0: No pretrain, 1: Pretrain with the learned embeddings')
    parser.add_argument('--pretrain_model_path', nargs='?', default='trained_model/model.pth', help='Path of stored model.')

    parser.add_argument('--DDI_batch_size', type=int, default=4096, help='DDI batch size.')  # Reduced from 64
    parser.add_argument('--kg_batch_size', type=int, default=4096, help='KG batch size.')  # Reduced from 64
    parser.add_argument('--DDI_evaluate_size', type=int, default=4096, help='DDI evaluate size.')  # Reduced from 64
    #parser.add_argument('-n', '--negative_sample_size', default=128, type=int)

    parser.add_argument('--entity_dim', type=int, default=256, help='Entity Embedding size.')
    parser.add_argument('--relation_dim', type=int, default=256, help='Relation Embedding size.')

    parser.add_argument('--aggregation_type', nargs='?', default='sum', help='Specify the type of the aggregation layer from {sum, concat, pna}.')
    parser.add_argument('--conv_dim_list', nargs='?', default='[64, 32, 16]', help='Output sizes of every aggregation layer.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1, 0.1]', help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')

    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5, help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--DDI_l2loss_lambda', type=float, default=1e-5, help='Lambda when calculating DDI l2 loss.')

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--stopping_steps', type=int, default=10, help='Number of epoch for early stopping')

    parser.add_argument('--ddi_print_every', type=int, default=50, help='Iter interval of printing DDI loss.')

    parser.add_argument('--kg_print_every', type=int, default=50, help='Iter interval of printing KG loss.')
    parser.add_argument('--evaluate_every', type=int, default=50, help='Epoch interval of evaluating DDI.')

    parser.add_argument('--multi_type', nargs='?', default='False', help='whether task is multi-class')
    parser.add_argument('--n_hidden_1', type=int, default=512, help='FC hidden 1 dim')  # Reduced from 2048
    parser.add_argument('--n_hidden_2', type=int, default=512, help='FC hidden 2 dim')  # Reduced from 2048
    parser.add_argument('--out_dim', type=int, default=1, help='FC output dim: 81 or 1')
    parser.add_argument('--structure_dim', type=int, default=300, help='structure_dim')
    parser.add_argument('--pre_entity_dim', type=int, default=200, help='pre_entity_dim')
    parser.add_argument('--feature_fusion', nargs='?', default='init_double', help='feature fusion type: concat / sum / init_double')

    args = parser.parse_args([])

    save_dir = BASE_DIR+'/workspace/Baseline/MUFFIN/results/250805/{}/all_entitydim{}_relationdim{}_feature{}_{}_{}_lr{}_pretrain{}/'.format(
        args.data_name, args.entity_dim, args.relation_dim, args.feature_fusion, args.aggregation_type,
        '-'.join([str(i) for i in eval(args.conv_dim_list)]), args.lr, args.use_pretrain)
    args.save_dir = save_dir

    return args

# %%
args = parse_SKGDDI_args()

# seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# GPU / CPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

# initialize data
data = DataLoaderSKGDDI(args)

n_approved_drug = data.n_approved_drug
n_entities = data.n_entities

# define pretrain embedding information
if args.use_pretrain == 1:

    if args.feature_fusion in ['sum', 'concat', 'init_double']:

        structure_pre_embed = torch.tensor(data.structure_pre_embed).to(device)
        entity_pre_embed = torch.tensor(data.entity_pre_embed).to(device).float()
        relation_pre_embed = torch.tensor(data.relation_pre_embed).to(device).float()
        embedding_pre = torch.LongTensor(range(data.n_approved_drug)).to(device)
        embedding_after = torch.LongTensor(range(data.n_approved_drug, data.n_entities)).to(device)

    else:
        entity_pre_embed, relation_pre_embed = None, None
        structure_pre_embed = torch.tensor(data.structure_pre_embed)

else:
    entity_pre_embed, relation_pre_embed, structure_pre_embed = None, None, None

train_graph = None

all_acc_list = []
all_precision_list = []
all_recall_list = []
all_f1_list = []
all_auc_list = []

all_macro_precision_list = []
all_macro_recall_list = []
all_macro_f1_list = []
all_micro_precision_list = []
all_micro_recall_list = []
all_micro_f1_list = []

# train model
# use 5-fold cross validation
for i in range(5):
    # Clear GPU cache before each fold
    torch.cuda.empty_cache()
    
    # construct model & optimizer
    model = GCNModel(args, data.n_entities, data.n_relations, entity_pre_embed, relation_pre_embed,
                        structure_pre_embed)
    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)

    # define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()  # For mixed precision training
    if args.multi_type != 'False':
        print('Multi classification')
        loss_func = torch.nn.CrossEntropyLoss()
    else:
        print('Binary classification')
        loss_func = torch.nn.BCEWithLogitsLoss()

    # define dataloader
    train_x = torch.from_numpy(data.DDI_train_data_X[i])
    train_y = torch.from_numpy(data.DDI_train_data_Y[i])
    test_x = torch.from_numpy(data.DDI_test_data_X[i])
    test_y = torch.from_numpy(data.DDI_test_data_Y[i])
    torch_dataset_train = Data.TensorDataset(train_x, train_y)
    torch_dataset_test = Data.TensorDataset(test_x, test_y)

    loader_train = Data.DataLoader(
        dataset=torch_dataset_train,
        batch_size=data.ddi_batch_size,
        shuffle=True
    )
    loader_test = Data.DataLoader(
        dataset=torch_dataset_test,
        batch_size=args.DDI_evaluate_size,
        shuffle=True
    )

    data_idx = Data.TensorDataset(torch.LongTensor(range(n_approved_drug)))
    loader_idx = Data.DataLoader(
        dataset=data_idx,
        batch_size=16,  # Reduced from 32
        shuffle=False
    )
    best_epoch = -1

    epoch_list = []

    acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    auc_list = []

    macro_precision_list = []
    macro_recall_list = []
    macro_f1_list = []

    micro_precision_list = []
    micro_recall_list = []
    micro_f1_list = []

    init_step = 0

    print("Start training DDI + KG model on fold {}!".format(i + 1))
    time0 = time()
    for epoch in tqdm(range(1, args.n_epoch + 1)):
        model.train()

        time1 = time()
        ddi_total_loss = 0
        n_ddi_batch = data.n_ddi_train[i] // data.ddi_batch_size + 1

        for step, (batch_x, batch_y) in enumerate(loader_train):
            iter = step + 1
            time2 = time()

            if use_cuda:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

            # Use mixed precision training
            out = model('calc_ddi_loss', train_graph, batch_x, embedding_pre, embedding_after, loader_idx, epoch)

            if args.multi_type == 'False':
                out = out.squeeze(-1)
                loss = loss_func(out, batch_y.float())
            else:
                loss = loss_func(out, batch_y.long())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ddi_total_loss += loss.item()
    
            if (iter % args.ddi_print_every) == 0:
                print(
                    'DDI Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean '
                    'Loss {:.4f}'.format(
                        epoch, iter, n_ddi_batch, time() - time2, loss.item(), ddi_total_loss / iter))
        
        print(
            'DDI Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(
                epoch,
                n_ddi_batch,
                time() - time1,
                ddi_total_loss / n_ddi_batch))

        print('DDI + KG Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

        if args.multi_type == 'False':
            if (epoch % args.evaluate_every) == 0:
                # Clear cache before evaluation
                torch.cuda.empty_cache()
                time1 = time()
                with torch.no_grad():  # Use no_grad for evaluation to save memory
                    precision, recall, f1, acc, auc, all_embed = evaluate(args, model, train_graph, loader_test,
                                                                            embedding_pre,
                                                                            embedding_after, loader_idx, epoch)
                print(
                    'DDI Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision {:.4f} Recall {:.4f} F1 {:.4f} ACC '
                    '{:.4f} AUC {:.4f}'.format(
                        epoch, time() - time1, precision, recall, f1, acc, auc))

                epoch_list.append(epoch)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
                acc_list.append(acc)
                auc_list.append(auc)
                best_auc, should_stop = early_stopping(auc_list, args.stopping_steps)

                if should_stop:
                    index = auc_list.index(best_auc)
                    all_acc_list.append(acc_list[index])
                    all_auc_list.append(auc_list[index])
                    all_precision_list.append(precision_list[index])
                    all_recall_list.append(recall_list[index])
                    all_f1_list.append(f1_list[index])
                    print('Final DDI Evaluation: Precision {:.4f} Recall {:.4f} F1 {:.4f} ACC '
                                    '{:.4f} AUC {:.4f}'.format(precision, recall, f1, acc, auc))
                    break

                if auc_list.index(best_auc) == len(auc_list) - 1:
                    save_model(all_embed, model, args.save_dir, epoch, best_epoch)
                    print('Save model on epoch {:04d}!'.format(epoch))
                    best_epoch = epoch

                if epoch == args.n_epoch:
                    index = auc_list.index(best_auc)
                    all_acc_list.append(acc_list[index])
                    all_auc_list.append(auc_list[index])
                    all_precision_list.append(precision_list[index])
                    all_recall_list.append(recall_list[index])
                    all_f1_list.append(f1_list[index])
                    print('Final DDI Evaluation: Precision {:.4f} Recall {:.4f} F1 {:.4f} ACC '
                                    '{:.4f} AUC {:.4f}'.format(precision, recall, f1, acc, auc))
        else:
            if (epoch % args.evaluate_every) == 0:
                # Clear cache before evaluation
                torch.cuda.empty_cache()
                time1 = time()
                with torch.no_grad():  # Use no_grad for evaluation to save memory
                    macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1, acc, all_embed = evaluate(
                        args,
                        model,
                        train_graph,
                        loader_test,
                        embedding_pre,
                        embedding_after,
                        loader_idx,
                        epoch)
                print(
                    'DDI Evaluation: Epoch {:04d} | Total Time {:.1f}s | Macro Precision {:.4f} Macro Recall {:.4f} '
                    'Macro F1 {:.4f} Micro Precision {:.4f} Micro Recall {:.4f} Micro F1 {:.4f} ACC {:.4f}'.format(
                        epoch, time() - time1, macro_precision, macro_recall, macro_f1, micro_precision,
                        micro_recall,
                        micro_f1, acc))

                epoch_list.append(epoch)

                macro_precision_list.append(macro_precision)
                macro_recall_list.append(macro_recall)
                macro_f1_list.append(macro_f1)

                micro_precision_list.append(micro_precision)
                micro_recall_list.append(micro_recall)
                micro_f1_list.append(micro_f1)

                acc_list.append(acc)
                # auc_list.append(auc)
                best_acc, should_stop = early_stopping(acc_list, args.stopping_steps)

                if should_stop:
                    index = acc_list.index(best_acc)
                    all_acc_list.append(acc_list[index])
                    # all_auc_list.append(auc_list[index])

                    all_macro_precision_list.append(macro_precision_list[index])
                    all_macro_recall_list.append(macro_recall_list[index])
                    all_macro_f1_list.append(macro_f1_list[index])

                    all_micro_precision_list.append(micro_precision_list[index])
                    all_micro_recall_list.append(micro_recall_list[index])
                    all_micro_f1_list.append(micro_f1_list[index])

                    print('Final DDI Evaluation: Macro Precision {:.4f} Macro Recall {:.4f} '
                                    'Macro F1 {:.4f} Micro Precision {:.4f} Micro Recall {:.4f} Micro F1 {:.4f} ACC {:.4f}'.format(
                        macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1, acc))
                    break

                if acc_list.index(best_acc) == len(acc_list) - 1:
                    save_model(all_embed, model, args.save_dir, epoch, best_epoch)
                    print('Save model on epoch {:04d}!'.format(epoch))
                    best_epoch = epoch

                if epoch == args.n_epoch:
                    index = acc_list.index(best_acc)
                    all_acc_list.append(acc_list[index])
                    # all_auc_list.append(auc_list[index])

                    all_macro_precision_list.append(macro_precision_list[index])
                    all_macro_recall_list.append(macro_recall_list[index])
                    all_macro_f1_list.append(macro_f1_list[index])

                    all_micro_precision_list.append(micro_precision_list[index])
                    all_micro_recall_list.append(micro_recall_list[index])
                    all_micro_f1_list.append(micro_f1_list[index])

                    print('Final DDI Evaluation: Macro Precision {:.4f} Macro Recall {:.4f} '
                                    'Macro F1 {:.4f} Micro Precision {:.4f} Micro Recall {:.4f} Micro F1 {:.4f} ACC {:.4f}'.format(
                        macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1, acc))
    
    # Clear GPU memory after each fold
    del model
    torch.cuda.empty_cache()
                    
    break  # FIXME: remove this line to run all 5 folds


# summary
print(all_acc_list)
print(all_precision_list)
print(all_recall_list)
print(all_f1_list)
print(all_auc_list)
mean_acc = np.mean(all_acc_list)
mean_precision = np.mean(all_precision_list)
mean_recall = np.mean(all_recall_list)
mean_f1 = np.mean(all_f1_list)
mean_auc = np.mean(all_auc_list)
print('5-fold cross validation DDI Mean Evaluation: Precision {:.4f} Recall {:.4f} F1 {:.4f} ACC '
                '{:.4f} AUC {:.4f}'.format(mean_precision, mean_recall, mean_f1, mean_acc, mean_auc))


# %%
