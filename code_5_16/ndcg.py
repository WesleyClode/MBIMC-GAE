import numpy as np

def dev_step(tset, train_m, test_m, net, dataset, args, nd_possible_rating_values):
    """
    Evaluates model on a dev set
    """
    batch_size = 256
    #print("tset:",tset)
    user_te = np.array(list(tset.keys()))
    #print("user_te:",user_te)
    user_te2 = user_te[:, np.newaxis]
    #user_te2 = user_te

    ll = int(len(user_te) / batch_size) + 1

    recall50 = []
    recall100 = []
    recall200 = []
    ndcg50 = []
    ndcg100 = []
    ndcg200 = []

    for batch_num in range(int(ll)):
        print(batch_num/ll*100,"%")

        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, len(user_te))
        # u_batch 是每个batch中的一个对user的一个list
        u_batch = user_te2[start_index:end_index]
        # batch_users 是这个batch中user的个数
        batch_users = end_index - start_index

        num_user = train_m.shape[0]#user总数
        num_movie = train_m.shape[1]#item总数
        user_list = user_te[start_index:end_index]
        batch_rating_pairs = generate_pair(user_list, num_movie)
        batch_dec_graph = generate_dec_graph(batch_rating_pairs, num_user, num_movie).to(args.device)


        Two_Stage = False
        pred_ratings, reg_loss, user_out, movie_out, W = net(dataset.train_enc_graph, batch_dec_graph, dataset.user_feature, dataset.movie_feature, Two_Stage)
        #pred_ratings = th.softmax(pred_ratings, dim=1)
        #print("pred_rating",pred_ratings.shape)
        pred_ratings = pred_ratings.cpu().detach().numpy()
        
        #pred_argmax = np.argmax(pred_ratings, axis=1)
        
        pred_index = np.zeros_like(pred_ratings[:,0])

        for j in range(len(pred_index)):
            #pred_index[j][pred_argmax[j]] = 1
            pred_index[j] = pred_ratings[j][1]
        #print("pred_rating",pred_index[0:10])

        #real_pred_ratings = (th.softmax(pred_ratings, dim=1) * nd_possible_rating_values.view(1, -1)).sum(dim=1)
        #real_pred_ratings = (th.from_numpy(pred_index).to(args.device) * nd_possible_rating_values.view(1, -1)).sum(dim=1)
        real_pred_ratings = th.from_numpy(pred_index).to(args.device)
        print("real_pred_ratings", th.sum(real_pred_ratings>=1))
        u_b = user_te[start_index:end_index]
        
        real_pred_ratings = real_pred_ratings.cpu()
        #print("pred_shape:", real_pred_ratings.shape)
        pre = real_pred_ratings.reshape(batch_users, -1)
        #print("pred_shape:", pre.shape)
        #pre = np.reshape(real_pred_ratings, (batch_users, num_movie))
        pre = pre.detach().numpy()
        idx = np.zeros_like(pre, dtype=bool)
        idx[train_m[u_b].nonzero()] = True
        pre[idx] = -np.inf   

        recall = []   

        for kj in [50, 100, 200]:
            idx_topk_part = np.argpartition(-pre, kj, 1)
            # print pre[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :kj]]
            # print idx_topk_part
            pre_bin = np.zeros_like(pre, dtype=bool)
            pre_bin[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :kj]] = True

            # print pre_bin

            true_bin = np.zeros_like(pre, dtype=bool)
            true_bin[test_m[u_b].nonzero()] = True

            tmp = (np.logical_and(true_bin, pre_bin).sum(axis=1)).astype(np.float32)
            #print("tmp:",tmp)
            recall.append(tmp / np.minimum(kj, true_bin.sum(axis=1)))
            #print("recall:",tmp / np.minimum(kj, true_bin.sum(axis=1)))
            # print tmp

        #print("recall:",recall)
        ndcg = []

        for kj in [50, 100, 200]:
            # 获取前20个元素的大致序号
            idx_topk_part = np.argpartition(-pre, kj, 1)
            #print("pre:",pre.shape)
            # 
            #print("idx_topk_part[:, :kj]:",idx_topk_part[:, :kj])
            #获取每个用户对应的前20个预测的index
            topk_part = pre[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :kj]]
            #print("topk_part:",topk_part[0:2])
            idx_part = np.argsort(-topk_part, axis=1)
            # 将预测分数进行排序，从大到校输出index的值
            #print("idx_part:",idx_part[0:2])
            idx_topk = idx_topk_part[np.arange(end_index - start_index)[:, np.newaxis], idx_part]
            # 得到原来的序列中的对应index
            #print("idx_topk:",idx_topk[0:2])
            tp = np.log(2) / np.log(np.arange(2, kj + 2))

            test_batch = test_m[u_b]
            #print("test_batch:",test_batch)



            DCG = (test_batch[np.arange(batch_users)[:, np.newaxis], idx_topk].toarray() * tp).sum(axis=1)
            # 就只计算真实结果在预测结果中的第几号的dcg
            #print("tp:",tp)
            #print("DCG:",DCG)
            IDCG = np.array([(tp[:min(n, kj)]).sum()
                             for n in test_batch.getnnz(axis=1)])
            #print("IDCG:",np.array([(tp[:min(n, kj)]).sum()
            #                 for n in test_batch.getnnz(axis=1)]))
            
            ndcg.append(DCG / IDCG)
        #print("ndcg:",ndcg)
        recall50.append(recall[0])
        recall100.append(recall[1])
        recall200.append(recall[2])
        ndcg50.append(ndcg[0])
        ndcg100.append(ndcg[1])
        ndcg200.append(ndcg[2])
        
    
    recall50 = np.hstack(recall50)
    recall100 = np.hstack(recall100)
    recall200 = np.hstack(recall200)
    ndcg50 = np.hstack(ndcg50)
    ndcg100 = np.hstack(ndcg100)
    ndcg200 = np.hstack(ndcg200)
    print("recall50:",recall50[0:10])
    print("ndcg50:", ndcg50.shape)

    print("recall50:", np.mean(recall50), "ndcg50:",np.mean(ndcg50))
    print("recall100:",np.mean(recall100),"ndcg100:", np.mean(ndcg100))
    print("recall200:",np.mean(recall200), "ndcg200:",np.mean(ndcg200))
    #f1.write(str(np.mean(recall100)) + ' ' + str(np.mean(ndcg100)) + '\n')
    #f1.flush()

    return np.mean(recall50), np.mean(recall100), np.mean(recall200), np.mean(ndcg50), np.mean(ndcg100), np.mean(ndcg200)
