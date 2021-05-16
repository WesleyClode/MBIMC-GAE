"""Training GCMC model on the MovieLens data set.

The script loads the full graph to the training device.
"""
import os, time
import argparse
import logging
import random
import string
import pandas as pd
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from data import DataSetLoader
#from data_custom import DataSetLoader
from model import BiDecoder, GCMCLayer, MLPDecoder
from utils import get_activation, get_optimizer, torch_total_param_num, torch_net_info, MetricLogger
from utils import to_etype_name
from sklearn.metrics import ndcg_score

def sample_negative(ratings, sample_rate, item_set):
    """
    input: 
    1. training rating ::pd.frame
    2. sample number::int
    3. item_set:a set of item::set
    """
     #"""return all negative items & 100 sampled negative items"""
    interact_status = ratings.groupby('user_id')['movie_id'].apply(set).reset_index().rename(columns={'itemId': 'interacted_items'})
    #print(interact_status)
    #item_list = set(item_list)
    interact_status['negative_items'] = interact_status['movie_id'].apply(lambda x: item_set - x)
    #print(interact_status['negative_items'])
    interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, sample_rate))
    return interact_status[['user_id', 'negative_items', 'negative_samples']]

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self._act = get_activation(args.model_activation)
        self.encoder = nn.ModuleList()
        self.encoder.append(GCMCLayer(args.rating_vals,
                                 args.src_in_units,
                                 args.dst_in_units,
                                 args.gcn_agg_units,
                                 args.gcn_out_units,
                                 args.gcn_dropout,
                                 args.gcn_agg_accum,
                                 agg_act=self._act,
                                 share_user_item_param=args.share_param,
                                 device=args.device))
        self.gcn_agg_accum = args.gcn_agg_accum
        self.rating_vals = args.rating_vals
        self.device = args.device
        self.gcn_agg_units = args.gcn_agg_units
        self.src_in_units = args.src_in_units
        for i in  range(1, args.layers):
            if args.gcn_agg_accum == 'stack':
                gcn_out_units = args.gcn_out_units * len(args.rating_vals)
            else:
                gcn_out_units = args.gcn_out_units
            self.encoder.append(GCMCLayer(args.rating_vals,
                                        args.gcn_out_units,
                                        args.gcn_out_units,
                                        gcn_out_units,
                                        args.gcn_out_units,
                                        args.gcn_dropout - i*0.1,
                                        args.gcn_agg_accum,
                                        agg_act=self._act,
                                        share_user_item_param=args.share_param,
                                        ini = False,
                                        device=args.device))

        if args.decoder == "Bi":
            self.decoder = BiDecoder(in_units= args.gcn_out_units, #* args.layers,
                                     num_classes=len(args.rating_vals),
                                     num_basis=args.gen_r_num_basis_func)

            '''
            self.decoder2 = MLPDecoder(in_units= args.gcn_out_units * 2,
                                     num_classes=len(args.rating_vals),
                                     num_basis=args.gen_r_num_basis_func)
            '''
        elif args.decoder == "MLP":
            if args.loss_func == "CE":
                num_classes = len(args.rating_vals)
            else:
                num_classes = 1
            self.decoder = MLPDecoder(in_units= args.gcn_out_units * args.layers,
                                     num_classes=num_classes,
                                     num_basis=args.gen_r_num_basis_func)
        self.rating_vals = args.rating_vals
          
    def forward(self, enc_graph, dec_graph, ufeat, ifeat, Two_Stage = False):
        user_out = []
        movie_out = []
        for i in range(0, args.layers):
            user_o, movie_o = self.encoder[i](
                enc_graph,
                ufeat,
                ifeat,
                Two_Stage)
            if i == 0:
                user_out = user_o
                movie_out = movie_o
            else:
                user_out += user_o / float(i + 1)
                movie_out += movie_o /float(i + 1)
            #user_out.append(user_o)
            #movie_out.append(movie_o)
            ufeat = user_o
            ifeat = movie_o
        #pred_ratings = self.decoder2(dec_graph, th.cat([user_out[0], user_out[1]], 1), th.cat([movie_out[1], movie_out[0]], 1))
        #user_out = th.cat(user_out, 1)
        #movie_out = th.cat(movie_out, 1)
        #print("user_out:", user_out[0])
        #print("movie_out:", movie_out[0])
        
        pred_ratings = self.decoder(dec_graph, user_out, movie_out)
        W_r_last = None
        reg_loss = 0.0
        '''
        for rating in self.rating_vals:
            rating = to_etype_name(rating)
            if W_r_last is not None:
                reg_loss += th.sum((self.encoder[0].W_r[rating] - W_r_last)**2)
            W_r_last = self.encoder[0].W_r[rating]
            #W_r_last_2 = self.encoder_2.W_r[rating]
        '''
        W = th.matmul(self.encoder[0].att, self.encoder[0].basis.view(self.encoder[0].basis_units, -1))
        W = W.view(len(self.rating_vals), self.src_in_units, -1)
        for i, rating in enumerate(self.rating_vals):
            rating = to_etype_name(rating)
            if i != 0:
                reg_loss += -th.sum(th.cosine_similarity(W[i,:,:], W[i-1,:,:], dim=1))
        return pred_ratings, reg_loss, user_out, movie_out, W

def evaluate(args, net, dataset, segment='valid', debug = False, idx = 0):
    possible_rating_values = dataset.possible_rating_values
    nd_possible_rating_values = th.FloatTensor(possible_rating_values).to(args.device)
    if segment == "valid":
        rating_values = dataset.valid_truths
        enc_graph = dataset.valid_enc_graph
        dec_graph = dataset.valid_dec_graph
    elif segment == "test":
        rating_values = dataset.test_truths
        enc_graph = dataset.test_enc_graph
        dec_graph = dataset.test_dec_graph
        user_map = dataset.global_user_id_map
        movie_map = dataset.global_movie_id_map
    else:
        raise NotImplementedError

    # Evaluate RMSE
    net.eval()
    with th.no_grad():
        pred_ratings, reg_loss, user_out, movie_out, W = net(enc_graph, dec_graph,
                           dataset.user_feature, dataset.movie_feature)
        #print(pred_ratings)
    if args.loss_func == "CE":
        max_rating, max_indices = th.max(pred_ratings, dim=1)
        pred = nd_possible_rating_values[max_indices]
        real_pred_ratings = (th.softmax(pred_ratings, dim=1) *
                            nd_possible_rating_values.view(1, -1)).sum(dim=1)
        num_correct = th.eq(pred, rating_values).sum().float().item()
        #print(float(num_correct) / float(rating_values.shape[0]))
    elif args.loss_func == "MLP":
        real_pred_ratings = pred_ratings[:, 0]
    rmse = ((real_pred_ratings - rating_values) ** 2.).mean().item()
    rmse = np.sqrt(rmse)
    return rmse

def evaluate_metric(args, net, dataset, segment='valid', debug = False):
    # 输入的是valid 与 test data里面的数据，每个人一个购买的数据，我们对预测的分进行排序，看testvalid的NDCG是多少
    # 我们可以对用户进行128一个batch进行计算
    # 对test data
    # 我们的pred 是对所有的用户进行预测，难免时间有点长，我们
    # 我们根据的是用户对所有物品的一个评分，这个评分对应一个物品，而我们的testdata里也有对应物品，我们看能不能预测到
    # input: predicted value\test data 
    # output: NDCG,recall
    possible_rating_values = dataset.possible_rating_values
    nd_possible_rating_values = th.FloatTensor(possible_rating_values).to(args.device)
    if segment == "test":
        rating_matrix = dataset.test_rating_matrix
        enc_graph = dataset.test_enc_graph
        dec_graph = dataset.test_recall_dec_graph
        user_len = len(list(pd.unique(dataset.test_rating_info["user_id"])))
    elif segment == "valid":
        rating_matrix = dataset.valid_rating_matrix
        enc_graph = dataset.valid_enc_graph
        dec_graph = dataset.valid_recall_dec_graph        
        user_len = len(list(pd.unique(dataset.valid_rating_info["user_id"])))
    else:
        raise NotImplementedError   

    # Evaluate RMSE
    net.eval()
    with th.no_grad():
        pred_ratings, reg_loss, user_out, movie_out, W = net(enc_graph, dec_graph, dataset.user_feature, dataset.movie_feature)
        if args.loss_func == "CE":
            max_rating, max_indices = th.max(pred_ratings, dim=1)
            pred = nd_possible_rating_values[max_indices]
            real_pred_ratings = (th.softmax(pred_ratings, dim=1) *
                                nd_possible_rating_values.view(1, -1)).sum(dim=1)
        elif args.loss_func == "MLP":
            real_pred_ratings = pred_ratings[:, 0]

    
    pred = real_pred_ratings.cpu().numpy()
    
    predition = np.reshape(pred, (user_len, movie_out.shape[0]))
    print("pred:",predition[0:2],predition.shape)
    #pred = real_pred_ratings.cpu().numpy()[0:movie_out.shape[0]]

    rating_matrix = rating_matrix.cpu().numpy()

    metric_ndcg = []

    ndcg_20 = ndcg_score(rating_matrix, predition, k=20)
    ndcg_40 = ndcg_score(rating_matrix, predition, k=40)
    ndcg_80 = ndcg_score(rating_matrix, predition, k=80)
    
    metric_ndcg.append(ndcg_20)
    metric_ndcg.append(ndcg_40)
    metric_ndcg.append(ndcg_80)

    if segment == "test":
        print("ndcg@20:",ndcg_20)
        print("ndcg@40:",ndcg_40)
        print("ndcg@80:",ndcg_80)

    return metric_ndcg    

def dev_step(args, net, dataset, segment='valid', debug = False):
    # 输入的是valid 与 test data里面的数据，每个人一个购买的数据，我们对预测的分进行排序，看testvalid的NDCG是多少
    # 我们可以对用户进行128一个batch进行计算
    # 对test data
    # 我们的pred 是对所有的用户进行预测，难免时间有点长，我们
    # 我们根据的是用户对所有物品的一个评分，这个评分对应一个物品，而我们的testdata里也有对应物品，我们看能不能预测到
    # 我们的data是这样的 就是taobao数据，一个用户有多种操作行为，最后的test set valid set 为倒数第一次与倒数第二次购买的物品
    # 我们可以对pred data想、进行一个batch的操作 总共有12000的人每128一个batch
    # input: predicted value\test data  
    # output: NDCG,recall
    # To do：1.train_m, test_m
    possible_rating_values = dataset.possible_rating_values
    nd_possible_rating_values = th.FloatTensor(possible_rating_values).to(args.device)
    if segment == "test":
        rating_matrix = dataset.test_rating_matrix
        enc_graph = dataset.test_enc_graph
        dec_graph = dataset.test_recall_dec_graph
        #dec_graph = dataset.test_dec_graph
        # print(dec_graph)
        # print(enc_graph)
        # print(dataset.train_enc_graph)
        # print(dataset.train_dec_graph)
        user_len = len(list(pd.unique(dataset.test_rating_info["user_id"])))
    elif segment == "valid":
        rating_matrix = dataset.valid_rating_matrix
        enc_graph = dataset.valid_enc_graph
        dec_graph = dataset.valid_recall_dec_graph        
        user_len = len(list(pd.unique(dataset.valid_rating_info["user_id"])))
    else:
        raise NotImplementedError   

    # Evaluate RMSE
    # 我们的encode是全体的数据 decode是全体的数据
    net.eval()
    with th.no_grad():
        pred_ratings, reg_loss, user_out, movie_out, W = net(enc_graph, dec_graph, dataset.user_feature, dataset.movie_feature)
        if args.loss_func == "CE":
            # 这是一种加权算最后分数的方法，也许效果不好，但是先用着
            max_rating, max_indices = th.max(pred_ratings, dim=1)
            pred = nd_possible_rating_values[max_indices]
            real_pred_ratings = (th.softmax(pred_ratings, dim=1) *
                                nd_possible_rating_values.view(1, -1)).sum(dim=1)

        elif args.loss_func == "MLP":
            real_pred_ratings = pred_ratings[:, 0]

    train_m = dataset.train_m
    test_m = dataset.test_m
    pred = real_pred_ratings.cpu().numpy()
    print("pred.shape:",pred.shape)
    print(pred)
    
    predition = np.reshape(pred, (user_len, movie_out.shape[0]))
    #predition = np.reshape(pred, (user_len, 39493))
    user_te = np.array(list(user_out.shape[0]))
    user_te2 = user_te[:, np.newaxis]
    ll = int(len(user_te) / 128) + 1

    # pre 对应的是decoder的输出，每个人对每个物品的一个打分
    recall50 = []
    recall100 = []
    recall200 = []
    ndcg50 = []
    ndcg100 = []
    ndcg200 = []

    test_recall_pair = self._generate_pair_value_for_recall_new(user_list, item_len)
    test_recall_dec_graph = self._generate_dec_graph(test_recall_pair)

    for batch_num in range(ll):

        start_index = batch_num * 128
        end_index = min((batch_num + 1) * 128, len(user_te))
        u_batch = user_te2[start_index:end_index]

        batch_users = end_index - start_index

        # feed_dict = {
        #     deep.input_u: u_batch,
        #     deep.dropout_keep_prob: 1.0,
        # }

        # pre = sess.run(
        #     deep.pre, feed_dict)
        # print("pre:",pre.shape)


        pre = predition[start_index:end_index]

        u_b = user_te[start_index:end_index]

        #pre = np.array(pre)
        #pre = np.delete(pre, -1, axis=1)
        
        # 让train已经有数据的地方为负无穷
        # 现在有一个想法就是也造两个这样的稀疏矩阵
        idx = np.zeros_like(pre, dtype=bool)
        idx[train_m[u_b].nonzero()] = True
        pre[idx] = -np.inf
        #print("pre:",pre.shape)
        # recall

        recall = []

        for kj in [20, 40, 80]:
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

        # ndcg10
        ndcg = []

        for kj in [20, 40, 80]:
            # 获取前20个元素的大致序号
            idx_topk_part = np.argpartition(-pre, kj, 1)
            print("pre:",pre.shape)
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
            print("tp:",tp)
            print("DCG:",DCG)
            IDCG = np.array([(tp[:min(n, kj)]).sum()
                             for n in test_batch.getnnz(axis=1)])
            print("IDCG:",np.array([(tp[:min(n, kj)]).sum()
                             for n in test_batch.getnnz(axis=1)]))
            
            ndcg.append(DCG / IDCG)

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
    
    if segment == "test":
        print("recall50:",recall50[0:10])
        print("ndcg50:", ndcg50.shape)

        print("recall50:", np.mean(recall50), "ndcg50:",np.mean(ndcg50))
        print("recall100:",np.mean(recall100),"ndcg100:", np.mean(ndcg100))
        print("recall200:",np.mean(recall200), "ndcg200:",np.mean(ndcg200))

    """
    ***
    """
    
    # pred = real_pred_ratings.cpu().numpy()
    
    # predition = np.reshape(pred, (user_len, movie_out.shape[0]))
    # print("pred:",predition[0:2],predition.shape)
    # #pred = real_pred_ratings.cpu().numpy()[0:movie_out.shape[0]]

    # rating_matrix = rating_matrix.cpu().numpy()

    # metric_ndcg = []

    # ndcg_20 = ndcg_score(rating_matrix, predition, k=20)
    # ndcg_40 = ndcg_score(rating_matrix, predition, k=40)
    # ndcg_80 = ndcg_score(rating_matrix, predition, k=80)
    
    # metric_ndcg.append(ndcg_20)
    # metric_ndcg.append(ndcg_40)
    # metric_ndcg.append(ndcg_80)



    return metric_ndcg    

def dev_step1(tset, train_m, test_m):
    """
    Evaluates model on a dev set

    """
    print("tset:",tset)
    user_te = np.array(list(tset.keys()))
    #print("user_te:",user_te)
    user_te2 = user_te[:, np.newaxis]
    #user_te2 = user_te

    ll = int(len(user_te) / 128) + 1

    recall50 = []
    recall100 = []
    recall200 = []
    ndcg50 = []
    ndcg100 = []
    ndcg200 = []

    for batch_num in range(ll):

        start_index = batch_num * 128
        end_index = min((batch_num + 1) * 128, len(user_te))
        u_batch = user_te2[start_index:end_index]

        batch_users = end_index - start_index

        feed_dict = {
            deep.input_u: u_batch,
            deep.dropout_keep_prob: 1.0,
        }

        pre = sess.run(
            deep.pre, feed_dict)
        print("pre:",pre.shape)

        u_b = user_te[start_index:end_index]

        pre = np.array(pre)
        print("pre378:",pre.shape)
        pre = np.delete(pre, -1, axis=1)
        print("pre380:",pre.shape)

        idx = np.zeros_like(pre, dtype=bool)
        idx[train_m[u_b].nonzero()] = True
        pre[idx] = -np.inf
        #print("pre:",pre.shape)
        # recall

        recall = []

        for kj in [20, 40, 80]:
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

        # ndcg10
        ndcg = []

        for kj in [20, 40, 80]:
            # 获取前20个元素的大致序号
            idx_topk_part = np.argpartition(-pre, kj, 1)
            print("pre:",pre.shape)
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
    f1.write(str(np.mean(recall100)) + ' ' + str(np.mean(ndcg100)) + '\n')
    f1.flush()

    return loss

def train(args):
    print(args)
    dataset = DataSetLoader(args.data_name, args.device,
                use_one_hot_fea=args.use_one_hot_fea,
                symm=args.gcn_agg_norm_symm,
                test_ratio=args.data_test_ratio,
                valid_ratio=args.data_valid_ratio,
                sample_rate = args.sample_rate)
    print("Loading data finished ...\n")

    args.src_in_units = dataset.user_feature_shape[1]
    args.dst_in_units = dataset.movie_feature_shape[1]
    args.rating_vals = dataset.possible_rating_values

    ### build the net
    net = Net(args=args)
    net = net.to(args.device)
    nd_possible_rating_values = th.FloatTensor(dataset.possible_rating_values).to(args.device)
    rating_loss_net = nn.CrossEntropyLoss()
    learning_rate = args.train_lr
    optimizer = get_optimizer(args.train_optimizer)(net.parameters(), lr=learning_rate)
    print("Loading network finished ...\n")

    ### perpare training data
    train_gt_labels = dataset.train_labels
    train_gt_ratings = dataset.train_truths

    ### prepare the logger
    # train_loss_logger = MetricLogger(['iter', 'loss', 'rmse'], ['%d', '%.4f', '%.4f'],
    #                                  os.path.join(args.save_dir, 'train_loss%d.csv' % args.save_id))
    # valid_loss_logger = MetricLogger(['iter', 'rmse', "ndcg_20", "ndcg_40", "ndcg_80"], ['%d', '%.4f',  '%.4f',  '%.4f',  '%.4f'],
    #                                  os.path.join(args.save_dir, 'valid_loss%d.csv' % args.save_id))
    # test_loss_logger = MetricLogger(['iter', 'rmse', "ndcg_20", "ndcg_40", "ndcg_80"], ['%d', '%.4f',  '%.4f',  '%.4f',  '%.4f'],
    #                                 os.path.join(args.save_dir, 'test_loss%d.csv' % args.save_id))
    ### prepare the logger
    train_loss_logger = MetricLogger(['iter', 'loss', 'rmse'], ['%d', '%.4f', '%.4f'],
                                     os.path.join(args.save_dir, 'train_loss.csv'))
    valid_loss_logger = MetricLogger(['iter', 'rmse'], ['%d', '%.4f'],
                                     os.path.join(args.save_dir, 'valid_loss.csv'))
    test_loss_logger = MetricLogger(['iter', 'rmse'], ['%d', '%.4f'],
                                    os.path.join(args.save_dir, 'test_loss.csv'))
    ### declare the loss information
    best_valid_rmse = np.inf
    best_valid_ndcg = -np.inf
    best_test_ndcg = []
    no_better_valid = 0
    best_iter = -1
    count_rmse = 0
    count_num = 0
    count_loss = 0
    
    dataset.train_enc_graph = dataset.train_enc_graph.int().to(args.device)
    dataset.train_dec_graph = dataset.train_dec_graph.int().to(args.device)
    dataset.valid_enc_graph = dataset.train_enc_graph
    dataset.valid_dec_graph = dataset.valid_dec_graph.int().to(args.device)
    dataset.test_enc_graph = dataset.test_enc_graph.int().to(args.device)
    dataset.test_dec_graph = dataset.test_dec_graph.int().to(args.device)

    #dataset.valid_recall_dec_graph = dataset.valid_recall_dec_graph.to(args.device)
    #dataset.test_recall_dec_graph = dataset.test_recall_dec_graph.to(args.device)

    print("Start training ...")
    dur = []


    train_rating_pairs, train_rating_values = dataset._generate_pair_value(dataset.train_rating_info)
    
    # 首先需要对每个用户采样出他的0样本，这个操作做一次就可以了。
    # 其次每次从这些样本中随机抽取一些边作为0的边
    # def sample_negative(interact_status, sample_num, random_number):
    #     #"""return sample_num sampled negative items"""
    #     random.seed(random_number)
    #     interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, sample_num))
    #     return interact_status[['user_id', 'negative_samples']]

    # def update_encode_graph(dataset, train_rating_pairs, train_rating_values, sampled_data, seed):
    #     #train_rating_pairs, train_rating_values = dataset._generate_pair_value(dataset.train_rating_info)
    #     train_rating_pairs_zeros, train_rating_values_zeros = dataset._generate_pair_value_for_zero(sampled_data)    
    #     train_rating_pairs_new = (np.append(train_rating_pairs[0], train_rating_pairs_zeros[0]), np.append(train_rating_pairs[1], train_rating_pairs_zeros[1]))
    #     train_rating_values_new = np.append(train_rating_values, train_rating_values_zeros)
    #     train_enc_graph_NS = dataset._generate_enc_graph(train_rating_pairs_new, train_rating_values_new, add_support = True)
    #     #print("dataset.train_dec_graph:", dataset.train_enc_graph)
    #     train_enc_graph_NS = train_enc_graph_NS.int().to(args.device)
    #     valid_enc_graph_NS = train_enc_graph_NS    
    #     return train_enc_graph_NS

    def update_encode_graph(dataset, train_rating_pairs, train_rating_values, sampled_data):
        train_rating_pairs_zeros, train_rating_values_zeros = dataset._generate_pair_value_for_zero(dataset.train_rating_info, sampled_data)
        train_rating_pairs = (np.append(train_rating_pairs[0], train_rating_pairs_zeros[0]), np.append(train_rating_pairs[1], train_rating_pairs_zeros[1]))
        train_rating_values = np.append(train_rating_values, train_rating_values_zeros)
        dataset.train_enc_graph = dataset._generate_enc_graph(train_rating_pairs, train_rating_values, add_support = True)
        dataset.train_enc_graph = dataset.train_enc_graph.int().to(args.device)
        dataset.valid_enc_graph = dataset.train_enc_graph    
        return dataset.train_enc_graph

    def sample_data(interact_status, random_number, sample_rate):
        random.seed(random_number)
        #print("length:", len(interact_status['negative_items']))
        #for i in interact_status['negative_items']
        #print("neg:\n",interact_status['negative_items'])
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, sample_rate))
        return interact_status[['user_id', 'negative_items', 'negative_samples']]


    seed_list = np.random.randint(0, 10000, (args.train_max_iter,))

    #negitive_all = dataset.negative_all(dataset.train_rating_info)
    # max_num = 0
    # for i in range(0,len(negitive_all)):
    #     if len(negitive_all['negative_items'][i]) > max_num:
    #         max_num = len(negitive_all['negative_items'][i])
    # min_num = np.inf
    # for i in range(0,len(negitive_all)):
    #     if len(negitive_all['negative_items'][i]) < min_num:
    #         min_num = len(negitive_all['negative_items'][i])

    # sheet = np.zeros((len(negitive_all), max_num))     
    # for i in range(0,len(negitive_all)):
    #     for j in range (0, len(np.array(negitive_all['negative_items'][i]))):
    #         sheet[i][j] =  np.array(negitive_all['negative_items'][i])[j]
    # sheet_new = sheet[:,:min_num]
    # print(sheet_new)
    # X = np.array(negitive_all['negative_items'])
    # max_len = max(len(xx) for xx in X) 
    # M = np.array( [np.concatenate([xx, np.zeros( max_len - len(xx))]) for xx in X])
    # sheet = []
    # for i in range(M.shape[0]):
    #     random.shuffle(M[i])
        #print(list(M[i]))
        #print(np.random.shuffle(list(M[i])))
        #sheet.append[np.random.shuffle(list(M[i]))]
    
    #np.random.randint(0,10,(4,3))

    # print("neg_all:",negitive_all)
    # sampled_data = sample_data(negitive_all, random_number = 1, sample_rate = 3)
    # dataset.train_enc_graph = update_encode_graph(dataset, train_rating_pairs, train_rating_values, sampled_data)
    # dataset.valid_enc_graph = dataset.train_enc_graph 

    for iter_idx in range(1, args.train_max_iter):
        # """
        # 方法：创建一个最基本的encode图，然后再加边，每次加一种点
        # """
        #print(len(negitive_all))
        #print ("max_num:", max_num,min_num)
        #print("M shape:", sheet_new.shape)
        #print("sheet:",M)
        #print(np.random.shuffle(np.array(sheet_new)))
        #map_matrix = np.random.randint(0,min_num,(sheet_new.shape[0], sheet_new.shape[1])) < 20
        #print(sheet_new[map_matrix].shape)
        #print(np.where(sheet_new[map_matrix]))
        #print(sheet_new)
        # if args.sample_rate > 0:
            # 这是随机采样的代码 
            # """
            # 如何采样？
            # 1. 单次采样：时间占用还好
            # 2. 每次随机采样：
            #     我们先存一个所有负样本的表，每次在这个负样本的表中去采样

            # 对于更新函数，我们需要对train_enc_graph进行更新，

            # 函数：
            # 1. 一个采样函数
            # 2. 更新函数
            # """
            # print(1)
            # sampled_data = sample_data(negitive_all, random_number = seed_list[iter_idx], sample_rate = 10)
            # print(2)
            # dataset.train_enc_graph = update_encode_graph(dataset, train_rating_pairs, train_rating_values, sampled_data)
            # print(3)
            # dataset.valid_enc_graph = dataset.train_enc_graph 
            #print(4)

        if iter_idx > 3:
            t0 = time.time()
        net.train()
        if iter_idx > 250:
            Two_Stage = True
        else:
            Two_Stage = False
        Two_Stage = False
        pred_ratings, reg_loss, user_out, movie_out, W = net(dataset.train_enc_graph, dataset.train_dec_graph,
                           dataset.user_feature, dataset.movie_feature, Two_Stage)
        #print("pre:",pred_ratings[0])
        if args.loss_func == "CE":
            loss = rating_loss_net(pred_ratings, train_gt_labels).mean() + args.ARR * reg_loss
            '''
            real_pred_ratings = (th.softmax(pred_ratings, dim=1) *
                                nd_possible_rating_values.view(1, -1)).sum(dim=1)
            mse_loss = th.sum((real_pred_ratings - train_gt_ratings) ** 2)
            loss += mse_loss * 0.0001
            '''
        elif args.loss_func == "Hinge":
            real_pred_ratings = (th.softmax(pred_ratings, dim=1) *
                                nd_possible_rating_values.view(1, -1)).sum(dim=1)
            gap = (real_pred_ratings - train_gt_labels) ** 2
            hinge_loss = th.where(gap > 1.0, gap*gap, gap).mean()
            loss = hinge_loss
        elif args.loss_func == "MSE":
            '''
            seeds = th.arange(pred_ratings.shape[0])
            random.shuffle(seeds)
            for i in range((pred_ratings.shape[0] - 1) // 50 + 1):
                start = i * 50
                end = (i + 1) * 50
                if end > (pred_ratings.shape[0] - 1):
                    end = pred_ratings.shape[0] - 1
                batch = seeds[start:end]
                loss = F.mse_loss(pred_ratings[batch, 0], nd_possible_rating_values[train_gt_labels[batch]]) + args.ARR * reg_loss
                count_loss += loss.item() * 50 / pred_ratings.shape[0]
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                #nn.utils.clip_grad_norm_(net.parameters(), args.train_grad_clip)
                optimizer.step()
                pred_ratings, reg_loss = net(dataset.train_enc_graph, dataset.train_dec_graph,
                                   dataset.user_feature, dataset.movie_feature)
            '''
            loss = th.mean((pred_ratings[:, 0] - nd_possible_rating_values[train_gt_labels]) ** 2) + args.ARR * reg_loss
        count_loss += loss.item()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(net.parameters(), args.train_grad_clip)
        optimizer.step()
        #print("iter:",iter_idx, loss)
        if iter_idx > 3:
            dur.append(time.time() - t0)

        if iter_idx == 1:
            print("Total #Param of net: %d" % (torch_total_param_num(net)))
            print(torch_net_info(net, save_path=os.path.join(args.save_dir, 'net%d.txt' % args.save_id)))

        if args.loss_func == "CE":
            real_pred_ratings = (th.softmax(pred_ratings, dim=1) *
                                nd_possible_rating_values.view(1, -1)).sum(dim=1)
        elif args.loss_func == "MSE":
            real_pred_ratings = pred_ratings[:, 0]

        rmse = ((real_pred_ratings - train_gt_ratings) ** 2).sum()
        count_rmse += rmse.item()
        count_num += pred_ratings.shape[0]

        if iter_idx % args.train_log_interval == 0:
            train_loss_logger.log(iter=iter_idx,
                                  loss=count_loss/(iter_idx+1), rmse=count_rmse/count_num)
            logging_str = "Iter={}, loss={:.4f}, rmse={:.4f}, time={:.4f}".format(
                iter_idx, count_loss/iter_idx, count_rmse/count_num,
                np.average(dur))
            count_rmse = 0
            count_num = 0
        
        
        
        if iter_idx % args.train_valid_interval == 0:
            valid_rmse = evaluate(args=args, net=net, dataset=dataset, segment='valid')
            valid_loss_logger.log(iter = iter_idx, rmse = valid_rmse)
            test_rmse = evaluate(args=args, net=net, dataset=dataset, segment='test')
            logging_str += ', Test RMSE={:.4f}'.format(test_rmse)
            test_loss_logger.log(iter=iter_idx, rmse=test_rmse)
            logging_str += ',\tVal RMSE={:.4f}'.format(valid_rmse)
            dev_step(args, net, dataset=dataset, segment='test', debug = False)

            if valid_rmse < best_valid_rmse:
                best_valid_rmse = valid_rmse
                no_better_valid = 0
                best_iter = iter_idx
                test_rmse = evaluate(args=args, net=net, dataset=dataset, segment='test', debug = True, idx = iter_idx)
                
                best_test_rmse = test_rmse
                test_loss_logger.log(iter=iter_idx, rmse=test_rmse)
                logging_str += ', Test RMSE={:.4f}'.format(test_rmse)
            else:
                no_better_valid += 1
                if no_better_valid > args.train_early_stopping_patience\
                    and learning_rate <= args.train_min_lr:
                    logging.info("Early stopping threshold reached. Stop training.")
                    break
                if no_better_valid > args.train_decay_patience:
                    new_lr = max(learning_rate * args.train_lr_decay_factor, args.train_min_lr)
                    if new_lr < learning_rate:
                        learning_rate = new_lr
                        logging.info("\tChange the LR to %g" % new_lr)
                        for p in optimizer.param_groups:
                            p['lr'] = learning_rate
                        no_better_valid = 0
        if iter_idx  % args.train_log_interval == 0:
            print(logging_str)
    print('Best Iter Idx={}, Best Valid RMSE={:.4f}, Best Test RMSE={:.4f}'.format(
        best_iter, best_valid_rmse, best_test_rmse))
    train_loss_logger.close()
    valid_loss_logger.close()
    test_loss_logger.close()

def config():
    parser = argparse.ArgumentParser(description='PGMC')
    parser.add_argument('--seed', default=125, type=int) #123
    parser.add_argument('--device', default='1', type=int,
                        help='Running device. E.g `--device 0`, if using cpu, set `--device -1`')
    parser.add_argument('--save_dir', type=str, help='The saving directory')
    parser.add_argument('--save_id', type=int, help='The saving log id')
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--data_name', default='yahoo_music', type=str,
                        help='The dataset name: ml-100k, ml-1m, ml-10m, flixster, douban, yahoo_music')
    parser.add_argument('--data_test_ratio', type=float, default=0.1) ## for ml-100k the test ration is 0.2
    parser.add_argument('--data_valid_ratio', type=float, default=0.05)
    parser.add_argument('--use_one_hot_fea', action='store_true', default=False)
    parser.add_argument('--model_activation', type=str, default="leaky")
    parser.add_argument('--sample_rate', type=int, default=1)
    parser.add_argument('--gcn_dropout', type=float, default=0.7)
    parser.add_argument('--gcn_agg_norm_symm', type=bool, default=True)
    parser.add_argument('--gcn_agg_units', type=int, default=1800)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")
    parser.add_argument('--gcn_out_units', type=int, default=75)
    parser.add_argument('--gen_r_num_basis_func', type=int, default=2)
    parser.add_argument('--train_max_iter', type=int, default=50000)
    parser.add_argument('--train_log_interval', type=int, default=1)
    parser.add_argument('--train_valid_interval', type=int, default=1)
    parser.add_argument('--train_optimizer', type=str, default="adam")
    parser.add_argument('--decoder', type=str, default="Bi")
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_min_lr', type=float, default=0.001)
    parser.add_argument('--train_lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--train_decay_patience', type=int, default=50)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--train_early_stopping_patience', type=int, default=200)
    parser.add_argument('--share_param', default=True, action='store_true')
    parser.add_argument('--ARR', type=float, default='0.000004')
    parser.add_argument('--loss_func', type=str, default='CE')
    parser.add_argument('--sparse_ratio', type=float, default=0.0)
    args = parser.parse_args()
    args.device = th.device(args.device) if args.device >= 0 else th.device('cpu')
    ### configure save_fir to save all the info

    now = int(round(time.time()*1000))
    now02 = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now/1000))
    if args.save_dir is None:
        args.save_dir = args.data_name+"_" + ''.join(now02)
    if args.save_id is None:
        args.save_id = np.random.randint(20)
    args.save_dir = os.path.join("log", args.save_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args

if __name__ == '__main__':
    '''
    ml_1m : param, ARR = 0.0000004, gcn_agg_units = 1000, gcn_agg_accum = sum, tmse = 0.8322, valid_ratio = 0.05
    ml_100k : param, ARR = 0.000001, gcn_agg_units = 500, gcn_agg_accum = sum, tmse = 0.9046, valid_ratio = 0.05
    1lyaer ml_1m : param, ARR = 0.0000005, gcn_agg_units = 2400, gcn_agg_accum = sum, tmse = 0.8305, valid_ratio = 0.05, gcn_out_units = 75
    1layer ml_100k : param, pos_emb, ARR = 0.000005, gcn_agg_units = 750, gcn_agg_accum = sum, tmse = 0.8974, valid_ratio = 0.05, gcn_out_units = 75
    2layer ml_100k : param, pos_emb, ARR = 0.000005, gcn_agg_units = 750, gcn_agg_accum = sum, tmse = 0.8969, valid_ratio = 0.05, gcn_out_units = 75
    2lyaer ml_1m : param, ARR = 0.0000004, gcn_agg_units = 1800, gcn_agg_accum = sum, tmse = 0.8319, valid_ratio = 0.05, gcn_out_units = 75
    '''
    args = config()
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)
    train(args)
