import numpy as np
from FRETboard.LSH.lsh import VectorsInLSH

def make_count_dict(x):
    """
    Turn np array of symbols into dict of form symbol: count
    """
    sc = np.unique(x, return_counts=True)
    return {s: c for s, c in zip(sc[0], sc[1])}


def get_lsh_gini_impurity(x, labels):
    """
    Take 2-D vector p samples of size q (p x q), return mean Gini impurity based on provided labels.
    """
    if type(labels) == list: labels = np.array(labels)
    bins = VectorsInLSH(8, x).search_results
    bin_symbols = np.unique(bins)
    unique_labels = np.unique(labels)
    lab_dicts = [make_count_dict(bins[labels == lab]) for lab in unique_labels]
    cor_dict = {mc: np.zeros(unique_labels.size) for mc in range(unique_labels.size)}
    for bs in bin_symbols:
        cur_counts = [cur_dict.get(bs, 0) for cur_dict in lab_dicts]
        mc = np.argmax(cur_counts)  # majority class
        cor_dict[mc] += cur_counts
    gini_list = []
    for cd in cor_dict:
        cur_sum = float(np.sum(cor_dict[cd]))
        gini_list.append(1 - np.sum([(cde / max(cur_sum, 1)) ** 2 for cde in cor_dict[cd]]))
    return np.mean(gini_list)

def get_lsh_diff_count(x, labels):
    """
    Take 2-D vector p samples of size q (p x q), find total difference between two labels in binned counts.
    works for binary labels.
    """
    bins = VectorsInLSH(8,x).search_results
    bin_symbols = np.unique(bins)
    dict_1 = make_count_dict(bins[labels])
    dict_2 = make_count_dict(bins[np.invert(labels)])
    d1_count = 0
    total_count_1 = 0
    for bs in bin_symbols:
        d1c = dict_1.get(bs,0)
        d2c = dict_2.get(bs,0)
        if d1c > d2c: 
            d1_count += d1c
            total_count_1 += d1c + d2c
    if total_count_1 == 0:
        return 0.5  # todo: solves the total_count == 1 problem but not elegant...
    return 1 - (( float(d1_count) / total_count_1) ** 2 + (( float(total_count_1 - d1_count) / total_count_1)) ** 2)

def lsh_classify(x_train, labels, x_new, min_purity=0.0, bits=8):
    """
    """
    unique_labels = np.unique(labels)
    x_all = np.vstack((x_train, x_new))

    bins_all = VectorsInLSH(bits, x_all).search_results
    bins_train, bins_new = np.split(bins_all, [x_train.shape[0]])
    unique_symbols = np.unique(bins_all)
    class_dict = dict()
    pred = np.repeat('unknown', x_new.shape[0])
    for us in unique_symbols:
        lb, cnt = np.unique(labels[bins_train == us], return_counts=True)
        if np.all(cnt == 0):
            continue
        best_label = lb[cnt == cnt.max()]
        if best_label.size != 1:
            continue
        if float(cnt.max()) / cnt.sum() < min_purity:
            continue
        best_label = best_label[0]
        if us in bins_new:
            pred[bins_new == us] = best_label
    return pred


def pick_top_minima(x, diff, w, n):
    x = np.copy(x)
    d = np.copy(diff)
    hw = w // 2
    out = []
    for _ in range(n):
        m = x[np.argsort(d)[0]]
        rm_idx = np.logical_or(x < m - hw, x > m + hw)
        x = x[rm_idx]
        d = d[rm_idx]
        out.append(m)
        if x.size == 0:
            break
    return out