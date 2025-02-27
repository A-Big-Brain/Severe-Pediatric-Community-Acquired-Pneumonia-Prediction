import numpy as np
import sklearn
import pandas as pd
import torch
from sklearn import metrics
import random
import string


path = '../'

# get the weight in loss function
def get_weig(ex_da, label_name_list, whe_binary):
    label = np.array(ex_da.loc[:, label_name_list])

    if whe_binary == 'binary':
        y = np.max(label, -1)
        class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
    elif whe_binary == 'nobinary':
        y = label
        num_positives = np.sum(y, 0)
        class_weights = (len(y) - num_positives) / num_positives

    class_weights = torch.tensor(class_weights, dtype=torch.float)

    return class_weights


# calculate met with Youden
def cal_acc_sen_spe(pred, lab, v):
    p_lab = pred.copy()
    p_lab[p_lab > v] = 1
    p_lab[p_lab <= v] = 0
    dif = 2 * lab - p_lab

    fn, tp, tn, fp = len(np.where(dif == 2)[0]), len(np.where(dif == 1)[0]), len(np.where(dif == 0)[0]), len(np.where(dif == -1)[0])
    acc = (tp + tn) / (fn + tp + tn + fp)
    sen = tp / (tp + fn + 1e-5)
    spe = tn / (tn + fp + 1e-5)

    return acc, sen, spe

# find best met
def find_best_met(pred, lab):
    cv = np.unique(pred)
    cv = np.sort(cv)
    n_cv = (np.array([cv[0] - 0.001] + list(cv)) + np.array(list(cv) + [cv[-1] + 0.001])) / 2

    met_li = []
    for v in n_cv:
        acc, sen, spe = cal_acc_sen_spe(pred, lab, v)
        met_li.append([acc, sen, spe, v, sen+spe])
    met_li = np.array(met_li)
    ind = np.argmax(met_li[:, 4])
    [b_acc, b_sen, b_spe, b_cut, _] = met_li[ind]

    return b_acc, b_sen, b_spe, b_cut

# calculate the performance in multiple labels
def cal_met(pred, lab):

    if len(list(lab.shape)) == 1:
        fpr, tpr, thresholds = metrics.roc_curve(lab, pred[:, 1], pos_label=1)
        Auc = metrics.auc(fpr, tpr)
        acc, sen, spe = cal_acc_sen_spe(pred[:, 1], lab, 0.5)
        met = np.array([Auc, acc, sen, spe])[np.newaxis, :]
    else:
        # auc
        auc_li = []
        for i in range(pred.shape[1]):
            fpr, tpr, thresholds = metrics.roc_curve(lab[:, i], pred[:, i], pos_label=1)
            Auc = metrics.auc(fpr, tpr)
            auc_li.append(Auc)

        # acc, sen, spe, and best cutoff value
        acc_li, sen_li, spe_li = [], [], []
        for i in range(pred.shape[1]):
            acc, sen, spe = cal_acc_sen_spe(pred[:, i], lab[:, i], 0.5)

            acc_li.append(acc)
            sen_li.append(sen)
            spe_li.append(spe)

        met = np.stack([auc_li, acc_li, sen_li, spe_li], axis=1)

    return met

# calculate performance with Youden index
def cal_met_wth_Youden(pred, lab):
    if len(list(lab.shape)) == 1:
        fpr, tpr, thresholds = metrics.roc_curve(lab, pred[:, 1], pos_label=1)
        Auc = metrics.auc(fpr, tpr)

        b_acc, b_sen, b_spe, b_cut = find_best_met(pred[:, 1], lab)
        met = np.array([Auc, b_acc, b_sen, b_spe, b_cut])[np.newaxis, :]
    else:
        # auc
        auc_li = []
        for i in range(pred.shape[1]):
            fpr, tpr, thresholds = metrics.roc_curve(lab[:, i], pred[:, i], pos_label=1)
            Auc = metrics.auc(fpr, tpr)
            auc_li.append(Auc)

        # acc, sen, spe, and best cutoff value
        acc_li, sen_li, spe_li, cut_of = [], [], [], []
        for i in range(pred.shape[1]):

            b_acc, b_sen, b_spe, b_cut = find_best_met(pred[:, i], lab[:, i])

            cv = np.unique(pred[:, i])
            cv = np.sort(cv)
            n_cv = (np.array([cv[0] - 0.001] + list(cv)) + np.array(list(cv) + [cv[-1] + 0.001])) / 2

            b_acc, b_sen, b_spe, b_cut = 0, 0, 0, 0
            for v in n_cv:
                acc, sen, spe = cal_acc_sen_spe(pred[:, i], lab[:, i], v)
                if sen + spe > b_sen + b_spe:
                    b_acc, b_sen, b_spe, b_cut = acc, sen, spe, v

            acc_li.append(b_acc)
            sen_li.append(b_sen)
            spe_li.append(b_spe)
            cut_of.append(b_cut)

        met = np.stack([auc_li, acc_li, sen_li, spe_li, cut_of], axis=1)

    return met

# combine string list
def com_mul_str(str_li):
    str_li = str_li + [''.join(random.sample(string.ascii_letters + string.digits, 4))]
    long_str = ''

    for s in str_li:
        long_str = long_str + '_' + str(s)

    return long_str[1:]

# check label name
def check_lab(args):
    # remove the invalid label
    lab_count1 = pd.read_excel(args.path + args.da_ty + '_tr_label.xlsx')[args.label_name_list].sum()
    lab_count2 = pd.read_excel(args.path + args.da_ty + '_te_label.xlsx')[args.label_name_list].sum()
    args.label_name_list = [x for x in args.label_name_list if lab_count1[x] > 0 and lab_count2[x] > 0 ]

    return args

