import math

import numpy as np
import torch
from torch import FloatTensor
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler

from typing import List





class ListNet(torch.nn.Module):
    def __init__(self, num_input_features: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # укажите архитектуру простой модели здесь
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_input_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, input_1: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_1)
        return logits


class Solution:
    def __init__(self, n_epochs: int = 5, listnet_hidden_dim: int = 30,
                 lr: float = 0.001, ndcg_top_k: int = 10):
        
        self._prepare_data()
        self.num_input_features = self.X_train.shape[1]
        self.ndcg_top_k = ndcg_top_k
        self.n_epochs = n_epochs

        self.model = self._create_model(
            self.num_input_features, listnet_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)


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
        self.ys_train = torch.FloatTensor(y_train)
        self.ys_test = torch.FloatTensor(y_test)



    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        
        for q in np.unique(inp_query_ids):
            idx = (inp_query_ids == q).nonzero()
            inp_feat_array[idx] = StandardScaler().fit_transform(inp_feat_array[idx])
        
        return inp_feat_array


    def _create_model(self, listnet_num_input_features: int,
                      listnet_hidden_dim: int) -> torch.nn.Module:
        torch.manual_seed(0)
        
        net = ListNet(
            num_input_features = listnet_num_input_features,
            hidden_dim = listnet_hidden_dim
        )
        return net
    

    def fit(self) -> List[float]:
        
        ndcg_list = []
        for _ in range(self.n_epochs):
            ndcg = self._train_one_epoch()
            ndcg_list.append(ndcg)

        return ndcg_list



    def _calc_loss(self, batch_ys: torch.FloatTensor,
                   batch_pred: torch.FloatTensor) -> torch.FloatTensor:
        
        P_y_i = torch.softmax(batch_ys, dim=0)
        P_z_i = torch.softmax(batch_pred, dim=0)
        return -torch.sum(P_y_i * torch.log(P_z_i/P_y_i))
    

    def _train_one_epoch(self) -> float:
        self.model.train()

        for q in np.unique(self.query_ids_train):
            
            idx = (self.query_ids_train == q).nonzero()

            batch_x = self.X_train[idx]
            batch_y = self.ys_train[idx]

            self.optimizer.zero_grad()
            
            if len(batch_x) > 0:

                batch_pred = self.model(batch_x).reshape(-1,)

                batch_loss = self._calc_loss(batch_y, batch_pred)
                print(f'LOSS: {batch_loss}')
                batch_loss.backward(retain_graph=True)

                self.optimizer.step()

        return self._eval_test_set()                    


    def _eval_test_set(self) -> float:
        with torch.no_grad():
            self.model.eval()
            ndcgs = []

            for q in np.unique(self.query_ids_test):
                idx = (self.query_ids_test == q).nonzero()
                batch_x = self.X_test[idx]
                batch_y = self.ys_test[idx]
                
                batch_pred = self.model(batch_x).reshape(-1,)

                ndcgs.append(self._ndcg_k(batch_y, batch_pred, ndcg_top_k=self.ndcg_top_k))

            return np.mean(ndcgs)

    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
                ndcg_top_k: int) -> float:
        
        ys_pred_k = ys_pred[:ndcg_top_k]
        ys_true_k = ys_true[:ndcg_top_k]


        ys_pred_ind = torch.argsort(ys_pred_k, descending=True, dim=0)
        ys_true_cons_sorted = ys_true_k[ys_pred_ind]

        dcg_k = 0
        for i, v in enumerate(ys_true_cons_sorted):
            dcg_k += gain(v)/math.log2(i+2)


        ys_true_sorted, _ = torch.sort(ys_true_k, descending=True, dim=0)

        ideal_dcg_k = 0
        for i, v in enumerate(ys_true_sorted):
            ideal_dcg_k += gain(v)/math.log2(i+2)
        
        if ideal_dcg_k:
            return dcg_k / ideal_dcg_k
        
        return 0
    
def gain(v: float):
    return 2 ** float(v) - 1


# solution = Solution(n_epochs=20)
# ndcg_list = solution.fit()
# print(ndcg_list)
