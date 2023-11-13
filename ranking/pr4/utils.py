import torch
import math

def compute_gain(y_value: float) -> float:
    return float(2 ** y_value - 1)

def dcg(ys_true: torch.Tensor, ys_pred: torch.Tensor,) -> float:
    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    ret = 0
    for idx, cur_y in enumerate(ys_true_sorted, 1):
        gain = compute_gain(cur_y)
        ret += gain / math.log2(idx + 1)
    return ret


def ndcg(ys_true: torch.Tensor, ys_pred: torch.Tensor) -> float:
    pred_dcg = dcg(ys_true, ys_pred)
    ideal_dcg = dcg(ys_true, ys_true)
    
    if ideal_dcg:
        return pred_dcg / ideal_dcg
    
    return 0

def compute_ideal_dcg(ys_true):
    return dcg(ys_true, ys_true)


def num_swapped_pairs(ys_true: torch.Tensor, ys_pred: torch.Tensor) -> int:
    ys_pred_sorted, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    
    num_objects = ys_true_sorted.shape[0]
    swapped_cnt = 0
    for cur_obj in range(num_objects - 1):
        for next_obj in range(cur_obj + 1, num_objects):
            if ys_true_sorted[cur_obj] < ys_true_sorted[next_obj]:
                if ys_pred_sorted[cur_obj] > ys_pred_sorted[next_obj]:
                    swapped_cnt += 1
            elif ys_true_sorted[cur_obj] > ys_true_sorted[next_obj]:
                if ys_pred_sorted[cur_obj] < ys_pred_sorted[next_obj]:
                    swapped_cnt += 1
    return swapped_cnt