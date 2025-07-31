#!/usr/bin/env python3
"""
Created on 2025-07-31 (Thu) 17:38:51

@author: I.Azuma
"""
# %%
import os
import copy
import numpy as np

from collections import OrderedDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import torch

def early_stopping(recall_list, stopping_steps):
    best_recall = max(recall_list)
    best_step = recall_list.index(best_recall)
    if len(recall_list) - best_step - 1 >= stopping_steps:
        should_stop = True
    else:
        should_stop = False
    return best_recall, should_stop

def save_model(all_embed, model, model_dir, current_epoch, last_best_epoch=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)

    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(last_best_epoch))
        if os.path.exists(old_model_state_file):
            os.system('rm {}'.format(old_model_state_file))

def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError:
        state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            k_ = k[7:]  # remove 'module.' of DistributedDataParallel instance
            state_dict[k_] = v
        model.load_state_dict(state_dict)

    model.eval()
    return model


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


def get_device(args):
    return torch.device('cpu') if args.gpu[0] < 0 else torch.device('cuda:' + str(args.gpu[0]))

# -------------------------------------- metrics and evaluation define -------------------------------------------------
def calc_metrics(y_true, y_pred, pred_score, multi_type):
    if multi_type != 'False':

        acc = accuracy_score(y_true, y_pred)
        macro_precision = precision_score(y_true, y_pred, average='macro')
        macro_recall = recall_score(y_true, y_pred, average='macro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_precision = precision_score(y_true, y_pred, average='micro')
        micro_recall = recall_score(y_true, y_pred, average='micro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        return acc, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1
    else:
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true.cuda().data.cpu().numpy(), pred_score.cuda().data.cpu().numpy())

        print(acc, precision, recall, f1, auc)

        return acc, precision, recall, f1, auc

def evaluate(args, model, train_graph, loader_test, embedding_pre, embedding_after, loader_idx, epoch):
    model.eval()

    precision_list = []
    recall_list = []
    f1_list = []
    acc_list = []
    auc_list = []

    macro_precision_list = []
    macro_recall_list = []
    macro_f1_list = []
    micro_precision_list = []
    micro_recall_list = []
    micro_f1_list = []

    with torch.no_grad():
        for data in loader_test:
            test_x, test_y = data
            out, all_embedding = model('predict', train_graph, test_x, embedding_pre, embedding_after, loader_idx,
                                       epoch)
            if args.multi_type == 'False':
                out = out.squeeze(-1)
                prediction = copy.deepcopy(out)
                prediction[prediction >= 0.5] = 1
                prediction[prediction < 0.5] = 0
                prediction = prediction.cuda().data.cpu().numpy()
                acc, precision, recall, f1, auc = calc_metrics(test_y, prediction, out, args.multi_type)
                acc_list.append(acc)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
                auc_list.append(auc)

            else:
                prediction = torch.max(out, 1)[1]
                prediction = prediction.cuda().data.cpu().numpy()
                acc, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = calc_metrics(
                    test_y,
                    prediction,
                    out,
                    args.multi_type)
                acc_list.append(acc)
                macro_precision_list.append(macro_precision)
                macro_recall_list.append(macro_recall)
                macro_f1_list.append(macro_f1)
                micro_precision_list.append(micro_precision)
                micro_recall_list.append(micro_recall)
                micro_f1_list.append(micro_f1)

    if args.multi_type == 'False':
        precision = np.mean(precision_list)
        recall = np.mean(recall_list)
        f1 = np.mean(f1_list)
        acc = np.mean(acc_list)
        auc = np.mean(auc_list)

        return precision, recall, f1, acc, auc, all_embedding
    else:
        macro_precision = np.mean(macro_precision_list)
        macro_recall = np.mean(macro_recall_list)
        macro_f1 = np.mean(macro_f1_list)
        micro_precision = np.mean(micro_precision_list)
        micro_recall = np.mean(micro_recall_list)
        micro_f1 = np.mean(micro_f1_list)
        acc = np.mean(acc_list)

        # print(acc, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1)

        return macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1, acc, all_embedding
