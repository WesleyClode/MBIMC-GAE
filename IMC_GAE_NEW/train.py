"""Training GCMC model on the MovieLens data set.

The script loads the full graph to the training device.
"""
import os, time
import argparse
import logging
import random
import string
import dgl
import scipy.sparse as sp
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

#f1 = open(os.path.join(DATA_ROOT, 'EHCF.txt'), 'w')

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

def generate_pair(user_list, num_movie):
    # 输入user_list num_movie
    # num_movie 是电影的总数
    rating_pairs = (np.array(np.array([[ele] * num_movie for ele in user_list]).flatten(),
                            dtype=np.int64),
                    np.array(np.array([[np.arange(num_movie)] * len(user_list)]).flatten(),
                            dtype=np.int64))
    return rating_pairs

def generate_dec_graph(rating_pairs, num_user, num_movie):
    #print(rating_pairs)
    #print("***:",len(rating_pairs), num_user, num_movie)
    ones = np.ones_like(rating_pairs[0])
    user_movie_ratings_coo = sp.coo_matrix(
        (ones, rating_pairs),
        shape=(num_user, num_movie), dtype=np.float32)
    g = dgl.bipartite_from_scipy(user_movie_ratings_coo, utype='_U', etype='_E', vtype='_V')
    return dgl.heterograph({('user', 'rate', 'movie'): g.edges()}, 
                            num_nodes_dict={'user': num_user, 'movie': num_movie})

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
    NDCG_logger = MetricLogger(['recall50', 'recall100', 'recall200','ndcg50', 'ndcg100', 'ndcg200'], ['%.4f', '%.4f', '%.4f','%.4f', '%.4f', '%.4f'], os.path.join(args.save_dir, 'NDCG.csv'))

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

    train_m = dataset.train_m
    test_m = dataset.test_m
    tset = dataset.tset

    user_num ,item_num = train_m.shape[0], train_m.shape[1]
    #dataset.valid_recall_dec_graph = dataset.valid_recall_dec_graph.to(args.device)
    #dataset.test_recall_dec_graph = dataset.test_recall_dec_graph.to(args.device)

    print("Start training ...")

    train_rating_pairs, train_rating_values = dataset._generate_pair_value(dataset.train_rating_info)
    
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
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, sample_rate))
        return interact_status[['user_id', 'negative_items', 'negative_samples']]


    seed_list = np.random.randint(0, 10000, (args.train_max_iter,))
    Two_Stage = False
    #sampled_data = sample_data(negitive_all, random_number = seed_list[iter_idx], sample_rate = 3)
    negitive_all = dataset.negative_all(dataset.train_rating_info)

    sampled_data = sample_data(negitive_all, random_number = 1, sample_rate = 99)
    dataset.train_enc_graph = update_encode_graph(dataset, train_rating_pairs, train_rating_values, sampled_data)
    dataset.valid_enc_graph = dataset.train_enc_graph 

    for iter_idx in range(1, args.train_max_iter):

        #sampled_data = sample_data(negitive_all, random_number = 1, sample_rate = 3)
        #dataset.train_enc_graph = update_encode_graph(dataset, train_rating_pairs, train_rating_values, sampled_data)
        
        print("iter:",iter_idx)
        net.train()
        pred_ratings, reg_loss, user_out, movie_out, W = net(dataset.train_enc_graph, dataset.train_dec_graph,
                           dataset.user_feature, dataset.movie_feature, Two_Stage)
        loss = rating_loss_net(pred_ratings, train_gt_labels).mean() + args.ARR * reg_loss
        count_loss += loss.item()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(net.parameters(), args.train_grad_clip)
        optimizer.step()
        real_pred_ratings = (th.softmax(pred_ratings, dim=1) * nd_possible_rating_values.view(1, -1)).sum(dim=1)
        #print(real_pred_ratings.shape)


        # 对pred的
        if iter_idx < 100:
            if iter_idx % 10 == 0:
                recall50_, recall100_, recall200_, ndcg50_, ndcg100_, ndcg200_ = \
                    dev_step(tset, train_m, test_m, net, dataset, args, nd_possible_rating_values)
                #dev_cold(u_train,i_train, tset, train_m, test_m)
                NDCG_logger.log(recall50 = recall50_, recall100 = recall100_, recall200 = recall200_, ndcg50 = ndcg50_, ndcg100 = ndcg100_, ndcg200 = ndcg200_)
            if iter_idx >= 500:
                recall50, recall100, recall200, ndcg50, ndcg100, ndcg200 = \
                    dev_step(tset, train_m, test_m, net, dataset, args ,nd_possible_rating_values)
                NDCG_logger.log(recall50 = recall50_, recall100 = recall100_, recall200 = recall200_, ndcg50 = ndcg50_, ndcg100 = ndcg100_, ndcg200 = ndcg200_)
            
                #dev_cold(u_train,i_train, tset, train_m, test_m)

        
    NDCG_logger.close()

def dev_step(tset, train_m, test_m, net, dataset, args, nd_possible_rating_values):
    """
    Evaluates model on a dev set
    """
    batch_size = 128
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

    for batch_num in range(ll):
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
        real_pred_ratings = (th.softmax(pred_ratings, dim=1) * nd_possible_rating_values.view(1, -1)).sum(dim=1)
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

        for kj in [20, 40, 80]:
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
