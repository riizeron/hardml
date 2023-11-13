import torch
import numpy as np

from math import log2
from torch import Tensor, sort


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:

    ys_pred_ind = torch.argsort(ys_pred, descending=True)
    ys_true_cons_sorted = ys_true[ys_pred_ind]

    ys_true_cons_sorted_pairs = torch.combinations(ys_true_cons_sorted)

    res = 0
    for n, k in ys_true_cons_sorted_pairs:
        res += 1 if n < k else 0

    return res


def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme == 'const':
       return y_value
    elif gain_scheme == 'exp2':
        return 2 ** y_value - 1


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    ys_pred_ind = torch.argsort(ys_pred, descending=True)
    ys_true_cons_sorted = ys_true[ys_pred_ind]

    i = 0
    res = 0
    for v in ys_true_cons_sorted:
        gain = compute_gain(float(v), gain_scheme)
        res += gain/log2(i+2)
        i += 1

    return res


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    dcg_v = dcg(ys_true, ys_pred, gain_scheme)

    ys_true_sorted, _ = torch.sort(ys_true, descending=True)
    i = 0
    dcg_i = 0
    for v in ys_true_sorted:
        gain = compute_gain(v, gain_scheme)
        dcg_i += gain/log2(i+2)
        i += 1
        
    return dcg_v / dcg_i


def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    l = ys_true.size(dim=0)
    k =  min(l, k)
    
    nonzero_count = torch.count_nonzero(ys_true)
    if not nonzero_count:
        return -1
        
    ys_pred_ind = torch.argsort(ys_pred, descending=True)
    ys_true_cons_sorted = ys_true[ys_pred_ind]

    nonzero_count_in_k = torch.count_nonzero(ys_true_cons_sorted[:k])

    return nonzero_count_in_k / min(k, nonzero_count)
    


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    l = ys_true.size(dim=0)

    ys_pred_ind = torch.argsort(ys_pred, descending=True)
    ys_true_cons_sorted = ys_true[ys_pred_ind]

    index = (ys_true_cons_sorted == 1).nonzero().squeeze()

    return 1 / (float(index) + 1)


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15 ) -> float:
    l = ys_true.size(dim=0)

    ys_pred_ind = torch.argsort(ys_pred, descending=True)
    ys_true_cons_sorted = ys_true[ys_pred_ind]

    res = 0
    p_look = 1
    for i in range(l):
        res += p_look * ys_true_cons_sorted[i]
        p_look = p_look * (1 - ys_true_cons_sorted[i]) * (1 - p_break)

    return float(res)

def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    l = ys_true.size(dim=0)
    
    nonzero_count = torch.count_nonzero(ys_true)
    if not nonzero_count:
        return -1
        
    ys_pred_ind = torch.argsort(ys_pred, descending=True)
    ys_true_cons_sorted = ys_true[ys_pred_ind]

    ones = 0
    res = 0
    for i in range(l):
        el = ys_true_cons_sorted[i]
        ones += el
        res += ones / (i+1) if el != 0.0 else 0

    return res / nonzero_count
