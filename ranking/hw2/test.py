import torch
import numpy as np

from solution_template import *


n = 5
task = 4

match task:
    case 1:
        #### 1

        y_true = torch.randperm(n)
        y_pred = torch.rand(n)

        print(f'y_pred: {y_pred}')
        print(f'y_true: {y_true}')

        res = num_swapped_pairs(y_true, y_pred)

        print(f'res: {res}')
    
    case 2:
        #### 2
        y_true = torch.tensor([3, 2, 1, 1, 3, 1, 2])
        y_pred = torch.tensor([3, 3, 2, 2, 1, 1, 1])

        dcg_v = dcg(y_true, y_pred, gain_scheme='const')
        ndcg_v = ndcg(y_true, y_pred, gain_scheme='const')
        print(f"DCG: {dcg_v}")
        print(f"NDCG: {ndcg_v}")

    case 4:
        # y_true = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0])
        # y_pred = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0])

        # res = precission_at_k(y_true, y_pred, k=12)
        # print(res)

        ys_true =  torch.tensor([0., 1., 1., 0., 1., 1., 0., 0., 0., 0.])
        ys_pred =  torch.tensor([0.2175, 0.3359, 0.0750, 0.7354, 0.4014, 0.2405, 0.2355, 0.4310, 0.2699, 0.0768])
        # Для K=7 ожидается 0.75
        res = precission_at_k(ys_true, ys_pred, k=7)
        print(res)

        # ys_true =  torch.tensor([1., 1., 1., 1., 0., 1., 0., 0., 0., 0.])
        # ys_pred =  torch.tensor([0.1060, 0.8329, 0.5542, 0.7974, 0.6109, 0.3831, 0.9206, 0.2759, 0.8032, 0.5047])
        # # Для K=8 ожидается 0.800000011920929
        # res = precission_at_k(ys_true, ys_pred, k=8)
        # print(res)

        # ys_true =  torch.tensor([1., 1., 1., 0., 0., 0., 0., 0., 0., 0.])
        # ys_pred =  torch.tensor([0.9, 0.8, 0.5, 0.1, 0.2, 0.3, 0.4, 0.12, 0.11, 0.01])
        # # Для K=8 ожидается 0.800000011920929
        # res = precission_at_k(ys_true, ys_pred, k=8)
        # print(res) 


# 1
# 1           k = 5
# 1
# 0
# 0
# 0



