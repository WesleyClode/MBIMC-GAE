"""Training GCMC model on the MovieLens data set.

The script loads the full graph to the training device.
"""
import os, time
import argparse
import logging
import random
import string
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
#import bottleneck as bn
from data import DataSetLoader
#from data_custom import DataSetLoader
from model import BiDecoder, GCMCLayer, MLPDecoder
from utils import get_activation, get_optimizer, torch_total_param_num, torch_net_info, MetricLogger
from utils import to_etype_name
from sklearn.metrics import ndcg_score


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
        for i in range(1, args.layers):
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
        #user_out = []
        #movie_out = []
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
            #print("user_out:", user_o[0])
            #print("movie_out:", movie_o[0])
        
        #pred_ratings = self.decoder2(dec_graph, th.cat([user_out[0], user_out[1]], 1), th.cat([movie_out[1], movie_out[0]], 1))
        #user_out = th.cat(user_out, 1)
        #movie_out = th.cat(movie_out, 1)
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
        #print("forward W:", W.shape)
        
        W = W.view(len(self.rating_vals), self.src_in_units, -1)
        for i, rating in enumerate(self.rating_vals):
            rating = to_etype_name(rating)
            if i != 0:
                reg_loss += -th.sum(th.cosine_similarity(W[i,:,:], W[i-1,:,:], dim=1))
        return pred_ratings, reg_loss, user_out, movie_out, W

def evaluate(args, net, dataset, segment='valid',debug = False, idx = 0):
    possible_rating_values = dataset.possible_rating_values
    nd_possible_rating_values = th.FloatTensor(possible_rating_values).to(args.device)
    if segment == "valid":
        rating_values = dataset.valid_truths
        user_index = dataset.test_rating_info["user_id"]
        enc_graph = dataset.valid_enc_graph
        dec_graph = dataset.valid_dec_graph
    elif segment == "test":
        rating_values = dataset.test_truths
        user_index = dataset.test_rating_info["user_id"]
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

    #print(dataset.user_feature)
    if args.loss_func == "CE":
        max_rating, max_indices = th.max(pred_ratings, dim=1)
        pred = nd_possible_rating_values[max_indices]
        real_pred_ratings = (th.softmax(pred_ratings, dim=1) *
                            nd_possible_rating_values.view(1, -1)).sum(dim=1)
        num_correct = th.eq(pred, rating_values).sum().float().item()

        print("correct rate:",float(num_correct) / float(rating_values.shape[0]))
    elif args.loss_func == "MLP":
        real_pred_ratings = pred_ratings[:, 0]
    rmse = ((real_pred_ratings - rating_values) ** 2.).mean().item()
    rmse = np.sqrt(rmse)
    #print("self.valid_labels:\n", dataset.valid_labels[0:10])
    #print("self.valid_truths:\n", dataset.valid_truths[0:10])
    
    '''
    if debug and segment == "test":
        fp = open("./debug/"+str(idx), "w")
        mse = (real_pred_ratings - rating_values) ** 2.
        edges = dec_graph.edges()
        for i in range(rating_values.shape[0]):
            print(str(user_map.inverse[edges[0][i].item()]) + " " + str(movie_map.inverse[edges[1][i].item()]) + " " + str(rating_values[i].item()) + " " + str(real_pred_ratings[i].item()) , file = fp)
        fp.close()
    '''
    return rmse

def evaluate_metric(args, net, dataset, segment='valid', debug = False):
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
        print("NDCG test")
        print("ndcg@20:",ndcg_20)
        print("ndcg@40:",ndcg_40)
        print("ndcg@80:",ndcg_80)

    return metric_ndcg    

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
    #args.decoder = "MLP"
    net = Net(args=args)
   
    #print(args)
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
    train_loss_logger = MetricLogger(['iter', 'loss', 'rmse'], ['%d', '%.4f', '%.4f'],
                                     os.path.join(args.save_dir, 'train_loss%d.csv' % args.save_id))
    valid_loss_logger = MetricLogger(['iter', 'rmse', "ndcg_20", "ndcg_40", "ndcg_80"], ['%d', '%.4f',  '%.4f',  '%.4f',  '%.4f'],
                                     os.path.join(args.save_dir, 'valid_loss%d.csv' % args.save_id))
    test_loss_logger = MetricLogger(['iter', 'rmse', "ndcg_20", "ndcg_40", "ndcg_80"], ['%d', '%.4f',  '%.4f',  '%.4f',  '%.4f'],
                                    os.path.join(args.save_dir, 'test_loss%d.csv' % args.save_id))

    ### declare the loss information
    best_valid_rmse = np.inf
    best_valid_ndcg = -np.inf
    best_test_ndcg = -np.inf
    no_better_valid = 0
    best_iter = -1
    count_rmse = 0
    count_num = 0
    count_loss = 0
    
    dataset.train_enc_graph = dataset.train_enc_graph.int().to(args.device)
    dataset.train_dec_graph = dataset.train_dec_graph.int().to(args.device)
    
    dataset.valid_enc_graph = dataset.train_enc_graph
    dataset.valid_dec_graph = dataset.valid_dec_graph.int().to(args.device)
    dataset.valid_recall_dec_graph = dataset.valid_recall_dec_graph.to(args.device)

    dataset.test_enc_graph = dataset.test_enc_graph.int().to(args.device)
    dataset.test_dec_graph = dataset.test_dec_graph.int().to(args.device)
    dataset.test_recall_dec_graph = dataset.test_recall_dec_graph.to(args.device)
    
    print("Start training ...")
    dur = []
    for iter_idx in range(1, args.train_max_iter):
        '''
        noisy_labels = th.LongTensor(np.random.choice([-1, 0, 1], train_gt_ratings.shape[0], replace=True, p=[0.001, 0.998, 0.001])).to(args.device)

        train_gt_labels += noisy_labels
    
        max_label = dataset.max_l + th.zeros_like(train_gt_labels)
        min_label = dataset.min_l + th.zeros_like(train_gt_labels)
        max_label = max_label.long()
        min_label = min_label.long()
        train_gt_labels = th.where(train_gt_labels > max_label, max_label, train_gt_labels)
        train_gt_labels = th.where(train_gt_labels < min_label, min_label, train_gt_labels)
        '''


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
        #print("user_out:\n", user_out[0])
        #print("movie_out:\n", movie_out[0])
        #print("W:\n", W.shape)
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
            train_loss_logger.log(iter=iter_idx, loss=count_loss/(iter_idx+1), rmse=count_rmse/count_num)
            logging_str = "Iter={}, loss={:.4f}, rmse={:.4f}, time={:.4f}".format(iter_idx, count_loss/iter_idx, count_rmse/count_num, np.average(dur))
            count_rmse = 0
            count_num = 0
        
        if iter_idx % args.train_valid_interval == 0:
            valid_rmse = evaluate(args=args, net=net, dataset=dataset, segment='valid')
            ndcg_valid = evaluate_metric(args=args, net=net, dataset=dataset, segment='valid', debug = False)
            print("ndcg_valid:",ndcg_valid)
            valid_loss_logger.log(iter = iter_idx, rmse = valid_rmse, ndcg_20 = ndcg_valid[0], ndcg_40 = ndcg_valid[1], ndcg_80 = ndcg_valid[2])
            print("-"*80)

            #test_rmse = evaluate(args=args, net=net, dataset=dataset, segment='test')
            #test_loss_logger.log(iter=iter_idx, rmse=test_rmse, ndcg_20 = ndcg_k[0], ndcg_40 = ndcg_k[1], ndcg_80 = ndcg_k[2])
            #logging_str += ', Test RMSE={:.4f}'.format(test_rmse)
            logging_str += ',\tVal RMSE={:.4f}'.format(valid_rmse)
            logging_str += ',\tndcg_valid_20={:.4f}'.format(ndcg_valid[0])
            logging_str += ',\tndcg_valid_40={:.4f}'.format(ndcg_valid[1])
            logging_str += ',\tndcg_valid_80={:.4f}'.format(ndcg_valid[2])
            
            ndcg_valid_20 = ndcg_valid[0]
            #print("***********",ndcg_valid_20)

            if ndcg_valid_20 > best_valid_ndcg:
                best_valid_ndcg = ndcg_valid_20
                print("************best_valid_ndcg:",best_valid_ndcg)
                print("************ndcg_valid_20:",ndcg_valid_20)
                no_better_valid = 0
                best_iter = iter_idx
                test_rmse = evaluate(args=args, net=net, dataset=dataset, segment='test', debug = True, idx = iter_idx)
                ndcg_test = evaluate_metric(args=args, net=net, dataset=dataset, segment ='test', debug = False)
                logging_str += ',\tbest ndcg_test={:.4f}'.format(ndcg_test[0])
                logging_str += ',\tbest ndcg_test={:.4f}'.format(ndcg_test[1])
                logging_str += ',\tbest ndcg_test={:.4f}'.format(ndcg_test[2])
                #best_test_rmse = test_rmse
                best_test_ndcg = ndcg_test
                #test_loss_logger.log(iter=iter_idx, rmse=test_rmse)
                test_loss_logger.log(iter=iter_idx, rmse=test_rmse, ndcg_20 = ndcg_test[0], ndcg_40 = ndcg_test[1], ndcg_80 = ndcg_test[2])
                #logging_str += ', Test RMSE={:.4f}'.format(test_rmse)
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
            #print("************best_valid_ndcg:",best_valid_ndcg)
            #print("************ndcg_valid_20:",ndcg_valid_20)            
        if iter_idx % args.train_log_interval == 0:
            print(logging_str)
    print('Best Iter Idx={}, best ndcg_20={:.4f}, best ndcg_40={:.4f}, best ndcg_80={:.4f}'.format(best_iter, best_test_ndcg[0], best_test_ndcg[1], best_test_ndcg[2]))
    train_loss_logger.close()
    valid_loss_logger.close()
    test_loss_logger.close()

def config():
    parser = argparse.ArgumentParser(description='PGMC')
    parser.add_argument('--seed', default=125, type=int) #123
    parser.add_argument('--device', default='0', type=int,
                        help='Running device. E.g `--device 0`, if using cpu, set `--device -1`')
    parser.add_argument('--save_dir', type=str, help='The saving directory')
    parser.add_argument('--save_id', type=int, help='The saving log id')
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--sample_rate', type=int, default=3)
    parser.add_argument('--data_name', default='yahoo_music', type=str,
                        help='The dataset name: ml-100k, ml-1m, ml-10m, flixster, douban, yahoo_music')
    parser.add_argument('--data_test_ratio', type=float, default=0.1) ## for ml-100k the test ration is 0.2
    parser.add_argument('--data_valid_ratio', type=float, default=0.05)
    parser.add_argument('--use_one_hot_fea', action='store_true', default=False)
    parser.add_argument('--model_activation', type=str, default="leaky")
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
    if args.save_dir is None:
        args.save_dir = args.data_name+"_" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=2))
    if args.save_id is None:
        args.save_id = np.random.randint(20)
    args.save_dir = os.path.join("log", args.save_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args

def ndcg(pred, gt, k):
    """
    Input???
    1. pred
    2. gt
    3. k
    k = min(k,len(pred))

    ??????pred??????????????????k?????????index1?????????gt[index1]
    ??????dcg(gt[index1])
    ??????gt???????????????k?????????index2?????????gt[index2]
    ??????dcg(gt[index2])

    return dcg(gt[index1])/dcg(gt[index2])
    """
    #print("pred:",pred.shape)
    #print("gt:",gt.shape)
    assert len(pred) == len(gt)

    k = min(k, len(pred)-1)
    # ?????????k?????????index????????????????????????????????????????????????????????????????????????????????????list???index
    #idx_topk1 = bn.argpartition(-pred, k)
    #print("idx_topk1:",idx_topk1)
    
    # 
    #topk_part = pred[idx_topk1]
    #print("topk_part:",topk_part)
    #idx_part = np.argsort(-topk_part)
    #print("idx_part:",idx_part)
    #f1 = topk_part[idx_part]
    #print("f1:",f1)
    
    index1 = np.argsort(-pred)[0:k+1]
    f1 = gt[index1]
    
    
    idx_topk2 = bn.argpartition(-gt, k)
    f2 = gt[idx_topk2]
    f2 = sorted(f2,reverse=True)
    #print("f1:",f1)
    #print("f2:",f2)

    return dcg(f1)/dcg(f2)

def dcg(scores):
    """
    Input???score
    Return: dcg

    ??????score??????????????????dcg
    """
    #sorted_scores = sorted(scores,reverse=True)
    k = len(scores)
    tp = 1. / np.log2(np.arange(2, k + 2))
    DCG = (scores * tp).sum(axis=0)

    return DCG

def comput_mean_ndcg(df, k):
    """
    Input???rating_info
            ???usr_id, movie_id, rating)
    output: ??????ndcg

    ??????????????? ???????????????????????????????????????dataframe
    ????????????values
    """
    #df.insert(df.shape[1], 'pred', pred)
    #print(df.groupby('user_id'))
    piece = dict(list(df.groupby('user_id')))
    ndcg_list = []
    for user in df["user_id"].unique():
        user_rating = piece[user]["rating"].values
        user_pred_rating = piece[user]['pred'].values
        ndcg_score = ndcg(user_pred_rating, user_rating, k)
        ndcg_list.append(ndcg_score)
    ndcg_list = np.array(ndcg_list)
    #return ndcg_list.mean()
    return ndcg_list

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
