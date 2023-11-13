
def listnet_ce_loss(y_i, z_i):
    """
    y_i: (n_i, 1) GT
    z_i: (n_i, 1) preds
    """

    P_y_i = torch.softmax(y_i, dim=0)
    P_z_i = torch.softmax(z_i, dim=0)
    return -torch.sum(P_y_i * torch.log(P_z_i))

def listnet_kl_loss(y_i, z_i):
    """
    y_i: (n_i, 1) GT
    z_i: (n_i, 1) preds
    """
    P_y_i = torch.softmax(y_i, dim=0)
    P_z_i = torch.softmax(z_i, dim=0)
    return -torch.sum(P_y_i * torch.log(P_z_i/P_y_i))



def _get_data() -> List[np.ndarray]:
    train_df, test_df = msrank_10k()

    X_train = train_df.drop([0, 1], axis=1).values
    y_train = train_df[0].values
    query_ids_train = train_df[1].values.astype(int)

    X_test = test_df.drop([0, 1], axis=1).values
    y_test = test_df[0].values
    query_ids_test = test_df[1].values.astype(int)

    return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]


def _prepare_data() -> None:
    (X_train, y_train, query_ids_train,
        X_test, y_test, query_ids_test) = _get_data()
    
    X_train = torch.FloatTensor(_scale_features_in_query_groups(X_train, query_ids_train))
    X_test = torch.FloatTensor(_scale_features_in_query_groups(X_test, query_ids_test))
    ys_train = torch.FloatTensor(y_train)
    ys_test = torch.FloatTensor(y_test)

    return X_train, ys_train, query_ids_train, X_test, ys_test, query_ids_test



def _scale_features_in_query_groups(inp_feat_array: np.ndarray,
                                    inp_query_ids: np.ndarray) -> np.ndarray:
    
    for q in np.unique(inp_query_ids):
        idx = (inp_query_ids == q).nonzero()
        inp_feat_array[idx] = StandardScaler().fit_transform(inp_feat_array[idx])
    
    return inp_feat_array



def make_dataset():
    return _prepare_data()



from pr4.utils import ndcg, num_swapped_pairs



def test(lr=0.01, n_epochs=5):
    X_train, ys_train, query_ids_train, X_test, ys_test, query_ids_test = make_dataset()
    num_input_features = X_train.shape[1]

    net = ListNet(num_input_features, hidden_dim=30)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    train_unique_ids = np.unique(query_ids_train)
    test_unique_ids = np.unique(query_ids_test)

    for epoch in range(n_epochs):

        for q in np.unique(train_unique_ids):
            
            idx = (query_ids_train == q).nonzero()
            print(idx)

            batch_X = X_train[idx]
            batch_ys = ys_train[idx]

            opt.zero_grad()
            if len(batch_X) > 0:
                batch_pred = net(batch_X)
                batch_loss = listnet_kl_loss(batch_ys, batch_pred)
    #             batch_loss = listnet_ce_loss(batch_ys, batch_pred)
                batch_loss.backward(retain_graph=True)
                opt.step()

            if (q - 1) % 150 == 0:
                with torch.no_grad():
                    ndcg_list = []
                    nsp_list = []
                    for q in np.unique(test_unique_ids):
            
                        idx = (query_ids_test == q).nonzero()

                        batch_X = X_test[idx]
                        batch_ys = ys_test[idx]

                        valid_pred = net(batch_X)
                        # print(f"valid_pred: {valid_pred}")
                        N_test = batch_X.shape[1]

                        valid_swapped_pairs = num_swapped_pairs(torch.flatten(batch_ys), torch.flatten(valid_pred))
                        ndcg_score = ndcg(batch_ys, valid_pred)

                        ndcg_list.append(ndcg_score)
                        nsp_list.append(valid_swapped_pairs)


                    
                    print(f"epoch: {epoch + 1}.\tNumber of swapped pairs: " 
                        f"{np.mean(nsp_list)}/{N_test * (N_test - 1) // 2}\t"
                        f"nDCG: {np.mean(ndcg_list)}")
                    
test()