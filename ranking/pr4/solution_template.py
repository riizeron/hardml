import math
import pickle
from typing import List, Tuple
import time

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import tqdm

# from hyperopt import fmin, tpe, hp, anneal, Trials


class Solution:
    def __init__(self, n_estimators: int = 100, lr: float = 0.5, ndcg_top_k: int = 10,
                 subsample: float = 0.6, colsample_bytree: float = 0.9,
                 max_depth: int = 5, min_samples_leaf: int = 8):
        self._prepare_data()

        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = int(n_estimators)
        self.lr = float(lr)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)

        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

        self.trees = []
        self.feature_ind = []
        self.best_ndcg = 0

        # допишите ваш код здесь

    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
            X_test, y_test, self.query_ids_test) = self._get_data()

        self.X_train = torch.FloatTensor(self._scale_features_in_query_groups(X_train, self.query_ids_train))
        self.X_test = torch.FloatTensor(self._scale_features_in_query_groups(X_test, self.query_ids_test))
        self.ys_train = torch.FloatTensor(y_train).reshape(-1,1)
        self.ys_test = torch.FloatTensor(y_test).reshape(-1,1)

        print(f'X_train: {self.X_train}')
        print(f'ys_train: {self.ys_train}')


    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:

        for q in np.unique(inp_query_ids):
            idx = (inp_query_ids == q).nonzero()
            inp_feat_array[idx] = StandardScaler().fit_transform(inp_feat_array[idx])
        
        return inp_feat_array

    def _train_one_tree(self, cur_tree_idx: int,
                        train_preds: torch.FloatTensor
                        ) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        
        sample_length = int(self.subsample * self.X_train.size(0))
        colsample_bytree_length = int(self.colsample_bytree * self.X_train.size(1))

        query_lambdas = torch.zeros_like(self.ys_train)
        for q in np.unique(self.query_ids_train):
            idx = (self.query_ids_train == q).nonzero()
                 
            query_lambdas[idx] = self._compute_lambdas(self.ys_train[idx], train_preds[idx])
        
        object_ind = torch.randperm(self.X_train.size(0))[:sample_length]
        feature_ind = torch.randperm(self.X_train.size(1))[:colsample_bytree_length]

        data = self.X_train[object_ind, :][:, feature_ind]

        tree = DecisionTreeRegressor(random_state=cur_tree_idx, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)

        tree.fit(data, query_lambdas[object_ind])

        return tree, feature_ind


    def _calc_data_ndcg(self, queries_list: np.ndarray,
                        true_labels: torch.FloatTensor, preds: torch.FloatTensor) -> float:

        with torch.no_grad():
            ndcgs = []

            for q in np.unique(queries_list):
                idx = (queries_list == q).nonzero()
                batch_x = preds[idx]
                batch_y = true_labels[idx]
                
                ndcgs.append(self._ndcg_k(batch_y, batch_x, ndcg_top_k=10))
            return np.mean(ndcgs)

    def fit(self):
        np.random.seed(0)

        pred_sum = torch.zeros_like(self.ys_train)
        max_ndcg_ind = 0


        for i in range(self.n_estimators):
            tree, f_ind = self._train_one_tree(i, pred_sum)
            
            tree_preds = torch.FloatTensor(tree.predict(self.X_train[:, f_ind])).reshape(-1,1)

            self.trees.append(tree)
            self.feature_ind.append(f_ind)

            ys_pred = self.predict(self.X_test)
            calc_ndcg = self._calc_data_ndcg(self.query_ids_test, self.ys_test, ys_pred)

            if calc_ndcg > self.best_ndcg:
                self.best_ndcg = calc_ndcg
                max_ndcg_ind = i

            print(f"{i}: NDCG CALC: {calc_ndcg}.")            

            pred_sum -= self.lr * tree_preds
        
        self.trees = self.trees[:max_ndcg_ind+1]


    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        res = torch.zeros_like(data[:, 0].reshape(-1,1), dtype=torch.float32)
        
        for q in np.unique(self.query_ids_test):
            idx = (q == self.query_ids_test).nonzero()
            q_X = data[idx]

            for t, f in zip(self.trees, self.feature_ind):
                res[idx] -= self.lr * torch.FloatTensor(t.predict(q_X[:,f])).reshape(-1,1)

        return res.float()


    def _compute_lambdas(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        ideal_dcg = compute_ideal_dcg(y_true)
        N = 1 / ideal_dcg if ideal_dcg != 0 else 0

        # рассчитаем порядок документов согласно оценкам релевантности
        _, rank_order = torch.sort(y_true, descending=True, axis=0)
        rank_order += 1
        
        with torch.no_grad():
            # получаем все попарные разницы скоров в батче
            pos_pairs_score_diff = 1.0 + torch.exp(y_pred - y_pred.t())
            
            # поставим разметку для пар, 1 если первый документ релевантнее
            # -1 если второй документ релевантнее
            Sij = compute_labels_in_batch(y_true)
            # посчитаем изменение gain из-за перестановок
            gain_diff = compute_gain_diff(y_true)
            
            # посчитаем изменение знаменателей-дискаунтеров
            decay_diff = (1.0 / torch.log2(rank_order + 1.0)) - (1.0 / torch.log2(rank_order.t() + 1.0))
            # посчитаем непосредственное иsзменение nDCG
            delta_ndcg = torch.abs(N * gain_diff * decay_diff)
            # посчитаем лямбды
            lambda_update =  (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
            lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)
            
            return lambda_update


    def _ndcg_k(self, ys_true, ys_pred, ndcg_top_k) -> float:

        ys_pred_k = ys_pred[:ndcg_top_k]
        ys_true_k = ys_true[:ndcg_top_k]
        
        ys_pred_ind = torch.argsort(torch.Tensor(ys_pred_k), descending=True, dim=0)
        ys_true_cons_sorted = ys_true_k[ys_pred_ind]

        dcg_k = 0
        for i, v in enumerate(ys_true_cons_sorted):
            dcg_k += gain(v)/math.log2(i+2)


        ys_true_sorted, _ = torch.sort(ys_true_k, descending=True, dim=0)

        ideal_dcg_k = 0
        for i, v in enumerate(ys_true_sorted):
            ideal_dcg_k += gain(v)/math.log2(i+2)
        
        if ideal_dcg_k:
            return float(dcg_k / ideal_dcg_k)
        
        return 0.


    def save_model(self, path: str):
        params = {
            'feature_ind': self.feature_ind,
            'trees': self.trees,
            'lr': self.lr
        }

        with open(path, 'wb') as f:
            pickle.dump(params, f)


    def load_model(self, path: str):
        with open(path, 'rb') as f:
            params = pickle.load(f)
            self.trees = params['trees']
            self.feature_ind = params['feature_ind']
            self.lr = params['lr']


def gain(x):
    return 2 ** x - 1
 

    
def compute_labels_in_batch(y_true: torch.Tensor):
    
    # разница релевантностей каждого с каждым объектом
    rel_diff = y_true - y_true.t()
    
    # 1 в этой матрице - объект более релевантен
    pos_pairs = (rel_diff > 0).type(torch.float32)
    
    # 1 тут - объект менее релевантен
    neg_pairs = (rel_diff < 0).type(torch.float32)
    Sij = pos_pairs - neg_pairs
    return Sij

def compute_gain_diff(y_true: torch.Tensor):
   return torch.pow(2.0, y_true) - torch.pow(2.0, y_true.t())

def compute_gain(y_value: float) -> float:
    return float(2 ** y_value - 1)

def dcg(ys_true: torch.Tensor, ys_pred: torch.Tensor, top=10) -> float:
    ys_pred = ys_pred[:top]
    ys_true = ys_true[:top]

    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    ret = 0
    for idx, cur_y in enumerate(ys_true_sorted, 1):
        gain = compute_gain(cur_y)
        ret += gain / math.log2(idx + 1)
    return ret


def ndcg(ys_true: torch.Tensor, ys_pred: torch.Tensor, top=10) -> float:
    pred_dcg = dcg(ys_true, ys_pred, top)
    ideal_dcg = dcg(ys_true, ys_true, top)
    
    if ideal_dcg:
        return float(pred_dcg / ideal_dcg)
    
    return 0.

def compute_ideal_dcg(ys_true, top=10):
    return dcg(ys_true, ys_true, top)

