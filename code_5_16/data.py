"""MovieLens dataset"""
import numpy as np
import os
import re
import pandas as pd
import scipy.sparse as sp
import torch as th
import scipy.sparse
from scipy.sparse import coo_matrix

import dgl
from dgl.data.utils import download, extract_archive, get_download_dir
from utils import to_etype_name

import pickle as pkl
import h5py
import pdb
import random
from scipy.sparse import linalg
from data_utils import load_data, map_data, download_dataset
from sklearn.metrics import mean_squared_error
from math import sqrt
from bidict import bidict

_urls = {
    'ml-100k' : 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'ml-1m' : 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
    'ml-10m' : 'http://files.grouplens.org/datasets/movielens/ml-10m.zip',
}

_paths = {
    'flixster' : './raw_data/flixster/training_test_dataset.mat',
    'douban' : './raw_data/douban/training_test_dataset.mat',
    'yahoo_music' : './raw_data/yahoo_music/training_test_dataset.mat',
    'ml-100k' : './raw_data/ml-100k/',
    'ml-1m' : './raw_data/ml-1m/',
    'ml-10m' : './raw_data/ml-10M100K/',
    'Tmall':'./raw_data/Tmall/tzzs_data.csv',
    'Tmall_small':'./raw_data/Tmall_small/Tmall_small.rating',
    'Tmall_0_4000_20_4000':'./raw_data/Tmall_small/Tmall_0_4000_20_4000.rating',
    'Tmall_20_4000_20_4000':'./raw_data/Tmall_small/Tmall_20_4000_20_4000.rating',
    'Tmall_40_4000_40_4000':'./raw_data/Tmall_small/Tmall_40_4000_40_4000.rating',
    "taobao_10_2":'./raw_data/taobao_10_2/',
    "taobao_15_5":'./raw_data/taobao_15_5/',
    "taobao_8_3":'./raw_data/taobao_8_3/',
    "taobao":'./raw_data/Taobao1/',
    "Beibei":'./raw_data/Beibei/'
    #'Tmall_small':'./raw_data/Tmall_small/tzzs_data.csv'
}

READ_DATASET_PATH = get_download_dir()
GENRES_ML_100K =\
    ['unknown', 'Action', 'Adventure', 'Animation',
     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
     'Thriller', 'War', 'Western']
GENRES_ML_1M = GENRES_ML_100K[1:]
GENRES_ML_10M = GENRES_ML_100K + ['IMAX']

def load_data(csv_file):
    tp = pd.read_csv(csv_file, sep='\t')
    return tp

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

class DataSetLoader(object):
    def __init__(self, name, device, mix_cpu_gpu=False,
                 use_one_hot_fea=True, symm=True,
                 test_ratio=0.1, valid_ratio=0.1,sparse_ratio = 0, sample_rate = 3):
        self._name = name
        self._device = device
        self._symm = symm
        self._test_ratio = test_ratio
        self._valid_ratio = valid_ratio
        print("_paths[self._name]:",_paths[self._name])
        self._dir = os.path.join(_paths[self._name])
        self.sample_rate = sample_rate
        print(self._name[0:5])
        if self._name in ['ml-100k', 'ml-1m', 'ml-10m']:
            # download and extract
            download_dir = get_download_dir()
            print("download_dir: ", download_dir)
            zip_file_path = '{}/{}.zip'.format(download_dir, name)
            download(_urls[name], path=zip_file_path)
            extract_archive(zip_file_path, '{}/{}'.format(download_dir, name))
            if name == 'ml-10m':
                root_folder = 'ml-10M100K'
            else:
                root_folder = name
            self._dir = os.path.join(download_dir, name, root_folder)
            print("Starting processing {} ...".format(self._name))
            self._load_raw_user_info()
            self._load_raw_movie_info()
            print('......')
            if self._name == 'ml-100k':
                self.all_train_rating_info = self._load_raw_rates(os.path.join(self._dir, 'u1.base'), '\t')
                self.test_rating_info = self._load_raw_rates(os.path.join(self._dir, 'u1.test'), '\t')
                self.all_rating_info = pd.concat([self.all_train_rating_info, self.test_rating_info])
            elif self._name == 'ml-1m' or self._name == 'ml-10m':
                self.all_rating_info = self._load_raw_rates(os.path.join(self._dir, 'ratings.dat'), '::')
                num_test = int(np.ceil(self.all_rating_info.shape[0] * self._test_ratio))
                shuffled_idx = np.random.permutation(self.all_rating_info.shape[0])
                self.test_rating_info = self.all_rating_info.iloc[shuffled_idx[: num_test]]
                self.all_train_rating_info = self.all_rating_info.iloc[shuffled_idx[num_test: ]]
            else:
                raise NotImplementedError
            print('......')
            num_valid = int(np.ceil(self.all_train_rating_info.shape[0] * self._valid_ratio))
            shuffled_idx = np.random.permutation(self.all_train_rating_info.shape[0])
            self.valid_rating_info = self.all_train_rating_info.iloc[shuffled_idx[: num_valid]]
            self.train_rating_info = self.all_train_rating_info.iloc[shuffled_idx[num_valid: ]]
            self.possible_rating_values = np.append(np.unique(self.train_rating_info["rating"].values) ,0)
        elif self._name in ['Tmall', 'Tmall_small'] or self._name[0:5] == 'Tmall':
            #self.all_rating_info, M = self._load_tmall(os.path.join(_paths[self._name]))
            #print(self._name[0:5])
            self.all_rating_info = self._load_raw_rates_Tmall(os.path.join(_paths[self._name]), ' ')
            #print(self.all_rating_info)
            num_test = int(np.ceil(self.all_rating_info.shape[0] * (1 - self._test_ratio)))
            shuffled_idx = np.random.permutation(self.all_rating_info.shape[0])
            #self.test_rating_info = self.all_rating_info.iloc[shuffled_idx[: num_test]]
            #self.all_train_rating_info = self.all_rating_info.iloc[shuffled_idx[num_test: ]]   
            self.test_rating_info = self.all_rating_info.iloc[num_test:]
            self.all_train_rating_info = self.all_rating_info.iloc[: num_test]
            #print("self.all_train_rating_info")
            #print(self.all_train_rating_info)


            user_list = pd.unique(self.all_rating_info["user_id"].values)
            item_list = pd.unique(self.all_rating_info["movie_id"].values)

            #print("*******", user_list)

            user_nodes, item_nodes = user_list, item_list
            print('......')
            num_valid = int(np.ceil(self.all_train_rating_info.shape[0] * self._valid_ratio))
            #shuffled_idx = np.random.permutation(self.all_train_rating_info.shape[0])
            #self.valid_rating_info = self.all_train_rating_info.iloc[shuffled_idx[: num_valid]]
            #self.train_rating_info = self.all_train_rating_info.iloc[shuffled_idx[num_valid: ]]
            self.valid_rating_info = self.all_train_rating_info.iloc[: num_valid]
            self.train_rating_info = self.all_train_rating_info.iloc[num_valid: ]
            shuffled_idx = np.random.permutation(self.train_rating_info.shape[0])
            self.train_rating_info = self.train_rating_info.iloc[shuffled_idx]

            self.possible_rating_values = np.append(np.unique(self.train_rating_info["rating"].values) ,0)
            #print(self.possible_rating_values)
        elif self._name in ['taobao', 'Beibei']:
            tp_test = load_data(os.path.join(self._dir, 'buy.test.txt'))
            tp_train = load_data(os.path.join(self._dir, 'buy.train.txt'))
            tp_view = load_data(os.path.join(self._dir, 'pv.csv'))
            tp_cart = load_data(os.path.join(self._dir, 'cart.csv'))
            tp_train.insert(tp_train.shape[1], 'rating', 1)
            tp_test.insert(tp_test.shape[1], 'rating', 1)
            tp_view = tp_view.drop(columns='time')
            tp_view = tp_view.drop(columns='count')
            tp_view.insert(tp_view.shape[1], 'rating', 1)
            tp_cart = tp_cart.drop(columns='time')
            tp_cart = tp_cart.drop(columns='count')
            tp_cart.insert(tp_cart.shape[1], 'rating', 2)
            
            # tp_train = tp_train[0:4429]
            # tp_test = tp_test[0:1000]
            # tp_view = tp_view[0:33084]
            # tp_cart = tp_cart[0:4329]

            colum = ['user_id','movie_id','rating']
            tp_train.columns = colum
            tp_test.columns = colum
            tp_view.columns = colum
            tp_cart.columns = colum

            tp_all = tp_train.append(tp_test)

            usercount, itemcount = get_count(tp_all, 'user_id'), get_count(tp_all, 'movie_id')

            n_users, n_items = usercount.shape[0], itemcount.shape[0]
            
            # n_users, n_items = usercount.shape[0], 39493
            # test buy data
            self.test_rating_info = tp_test
            
            self.test_rating_info.columns = colum
            #shuffled_idx = np.random.permutation(self.test_rating_info.shape[0])
            #self.test_rating_info = self.test_rating_info.iloc[shuffled_idx]
            #### valid buy data
            # data_list = tp_train.values
            # result = []
            # for i in range(0,  data_list.shape[0]):
            #     if data_list[i][0] == data_list[i-1][0]+1:
            #         result.append(data_list[i-1])
            # result = np.squeeze(result)
            # self.valid_rating_info = pd.DataFrame(result)
            
            # self.valid_rating_info.columns = colum
            # shuffled_idx = np.random.permutation(self.valid_rating_info.shape[0])
            # self.valid_rating_info = self.valid_rating_info.iloc[shuffled_idx]
            #### train buy data
            #data_list = tp_train.values
            # result = []
            # for i in range(0,  data_list.shape[0]-1):
            #     if data_list[i+1][0] == data_list[i][0]:
            #         result.append(data_list[i])
            # result = np.squeeze(result)
            # tp_train = pd.DataFrame(result)
            # tp_train = pd.DataFrame(data_list)
            # buy data add cart and view
            #frames = []
            # colum = ['user_id','movie_id','rating']
            # tp_train.columns = colum
            # tp_test.columns = colum
            # tp_view.columns = colum
            # tp_cart.columns = colum
            # 所有数据
            #self.all_train_rating_info = pd.concat([tp_train, tp_cart, tp_view],axis = 0)
            self.all_train_rating_info = pd.concat([tp_train],axis = 0)
            #self.all_train_rating_info = pd.concat([tp_train, tp_cart],axis = 0)
            #self.all_train_rating_info = tp_train
            self.all_train_rating_info.columns = colum
            #shuffled_idx = np.random.permutation(self.all_train_rating_info.shape[0])
            #self.all_train_rating_info = self.all_train_rating_info.iloc[shuffled_idx]

            # self.all_train_rating_info = self._load_raw_rates_taobao(os.path.join(self._dir, 'taobao_train.rating'), ' ')
            # print("rating:",self.all_train_rating_info)
            # self.test_rating_info = self._load_raw_rates_taobao(os.path.join(self._dir, 'taobao_test.rating'), ' ')
            # self.valid_rating_info = self._load_raw_rates_taobao(os.path.join(self._dir, 'taobao_valid.rating'), ' ')
            # self.all_rating_info = pd.concat([self.all_train_rating_info, self.test_rating_info, self.valid_rating_info])
            self.all_rating_info = pd.concat([self.all_train_rating_info, self.test_rating_info])
            #print("self.all_train_rating_info:",self.all_train_rating_info[0:10])
            print('......')
            self.train_rating_info = self.all_train_rating_info
            self.possible_rating_values = np.append(np.unique(self.train_rating_info["rating"].values) ,0)
            #self.possible_rating_values = np.unique(self.train_rating_info["rating"].values)
            user_list = pd.unique(self.all_rating_info["user_id"].values)
            item_list = pd.unique(self.all_rating_info["movie_id"].values)
            user_nodes, item_nodes = user_list, item_list


            u_train = np.array(tp_train['user_id'], dtype=np.int32)
            i_train = np.array(tp_train['movie_id'], dtype=np.int32)
            u_test = np.array(tp_test['user_id'], dtype=np.int32)
            i_test = np.array(tp_test['movie_id'], dtype=np.int32)

            u_view = np.array(tp_view['user_id'], dtype=np.int32)
            i_view = np.array(tp_view['movie_id'], dtype=np.int32)
            u_cart = np.array(tp_cart['user_id'], dtype=np.int32)
            i_cart = np.array(tp_cart['movie_id'], dtype=np.int32)
            
            print(u_train)
            count = np.ones(len(u_train))
            print("(count, (u_train, i_train):",(count.shape, (u_train.shape, i_train.shape)))
            print("(n_users, n_items):",(n_users, n_items))
            train_m = scipy.sparse.csr_matrix((count, (u_train, i_train)), dtype=np.int16, shape=(n_users, n_items))
            print("train_m:",train_m.shape)
            count = np.ones(len(u_test))
            
            test_m = scipy.sparse.csr_matrix((count, (u_test, i_test)), dtype=np.int16, shape=(n_users, n_items))
            print("test_m:",test_m.shape)

            tset = {}
            for i in range(len(u_test)):
                if u_test[i] in tset: 
                #if tset.has_key(u_test[i]):
                    tset[u_test[i]].append(i_test[i])
                else:
                    tset[u_test[i]] = [i_test[i]]
            
            self.tset = tset

            
            self.train_m = train_m
            self.test_m = test_m
            print('......')
        else:
            raise NotImplementedError


        self.user_poll = set(pd.unique(self.all_rating_info["user_id"].values))
        self.item_poll = set(pd.unique(self.all_rating_info["movie_id"].values))
        
        self.negatives = []
        #self.negatives = self.sample_negative(self.train_rating_info, self.sample_rate, random_number=1)

        print("All rating pairs : {}".format(self.all_rating_info.shape[0]))
        print("\tAll train rating pairs : {}".format(self.all_train_rating_info.shape[0]))
        print("\t\tTrain rating pairs : {}".format(self.train_rating_info.shape[0]))
        #print("\t\tValid rating pairs : {}".format(self.valid_rating_info.shape[0]))
        print("\tTest rating pairs : {}".format(self.test_rating_info.shape[0]))

        if self._name in ['ml-100k', 'ml-1m', 'ml-10m']:
            self.user_info = self._drop_unseen_nodes(orign_info=self.user_info,
                                                    cmp_col_name="id",
                                                    reserved_ids_set=set(self.all_rating_info["user_id"].values),
                                                    label="user")
            self.movie_info = self._drop_unseen_nodes(orign_info=self.movie_info,
                                                    cmp_col_name="id",
                                                    reserved_ids_set=set(self.all_rating_info["movie_id"].values),
                                                    label="movie")

            # Map user/movie to the global id
            self.global_user_id_map = {ele: i for i, ele in enumerate(self.user_info['id'])}
            self.global_movie_id_map = {ele: i for i, ele in enumerate(self.movie_info['id'])}
        elif self._name in ['flixster', 'douban', 'yahoo_music','Tmall','Tmall_small','taobao','Beibei'] or self._name[0:5] == 'Tmall' or self._name[0:6] == 'taobao':
            self.global_user_id_map = bidict({})
            self.global_movie_id_map = bidict({})
            # max_uid = 0
            # max_vid = 0
            print("user and item number:")
            # print(user_nodes)
            # print(item_nodes)
            for i in range(len(user_nodes)):
                self.global_user_id_map[user_nodes[i]] = i
            for i in range(len(item_nodes)):
                self.global_movie_id_map[item_nodes[i]] = i
        else:
            raise NotImplementedError

        print('Total user number = {}, movie number = {}'.format(len(self.global_user_id_map),
                                                                 len(self.global_movie_id_map)))
        self._num_user = len(self.global_user_id_map)
        self._num_movie = len(self.global_movie_id_map)
        ### Generate features
        if use_one_hot_fea:
            self.user_feature = None
            self.movie_feature = None
        else:
            raise NotImplementedError

        # if self.user_feature is None:
        #     self.user_feature_shape = (self.num_user, self.num_user + self.num_movie + 3)
        #     self.movie_feature_shape = (self.num_movie, self.num_user + self.num_movie + 3)
        #     if mix_cpu_gpu:
        #         self.user_feature = th.cat([th.Tensor(list(range(3, self.num_user+3))).reshape(-1, 1), th.zeros([self.num_user, 1])+1, th.zeros([self.num_user, 1])], 1)
        #         self.movie_feature = th.cat([th.Tensor(list(range(3, self.num_movie+3))).reshape(-1, 1), th.ones([self.num_movie, 1])+1, th.zeros([self.num_movie, 1])], 1)
        #         # self.movie_feature = th.cat([th.Tensor(list(range(self.num_user+3, self.num_user + self.num_movie + 3))).reshape(-1, 1), th.ones([self.num_movie, 1])+1, th.zeros([self.num_movie, 1])], 1)
        #     else:
        #         self.user_feature = th.cat([th.Tensor(list(range(3, self.num_user+3))).reshape(-1, 1), th.zeros([self.num_user, 1])+1, th.zeros([self.num_user, 1])], 1).to(self._device)
        #         self.movie_feature = th.cat([th.Tensor(list(range(self.num_user+3, self.num_user + self.num_movie + 3))).reshape(-1, 1), th.ones([self.num_movie, 1])+1, th.zeros([self.num_movie, 1])], 1).to(self._device)
        # else:
        #     raise NotImplementedError

        if self.user_feature is None:
            self.user_feature_shape = (self.num_user, self.num_user + self.num_movie + 3)
            self.movie_feature_shape = (self.num_movie, self.num_user + self.num_movie + 3)
            if mix_cpu_gpu:
                self.user_feature = th.cat([th.Tensor(list(range(3, self.num_user+3))).reshape(-1, 1)], 1)
                self.movie_feature = th.cat([th.Tensor(list(range(3, self.num_movie+3))).reshape(-1, 1)], 1)
                # self.movie_feature = th.cat([th.Tensor(list(range(self.num_user+3, self.num_user + self.num_movie + 3))).reshape(-1, 1), th.ones([self.num_movie, 1])+1, th.zeros([self.num_movie, 1])], 1)
            else:
                self.user_feature = th.cat([th.Tensor(list(range(3, self.num_user+3))).reshape(-1, 1)], 1).to(self._device)
                self.movie_feature = th.cat([th.Tensor(list(range(self.num_user+3, self.num_user + self.num_movie + 3))).reshape(-1, 1)], 1).to(self._device)
        else:
            raise NotImplementedError

        # print(self.user_feature.shape)
        info_line = "Feature dim: "
        info_line += "\nuser: {}".format(self.user_feature_shape)
        info_line += "\nmovie: {}".format(self.movie_feature_shape)
        print(info_line)
        #print(self.valid_rating_info)  
        all_train_rating_pairs, all_train_rating_values = self._generate_pair_value(self.all_train_rating_info)
        train_rating_pairs, train_rating_values = self._generate_pair_value(self.train_rating_info)
        #valid_rating_pairs, valid_rating_values = self._generate_pair_value(self.valid_rating_info)
        test_rating_pairs, test_rating_values = self._generate_pair_value(self.test_rating_info)
               
        def _make_labels(ratings):
            labels = th.LongTensor(np.searchsorted(self.possible_rating_values, ratings)).to(device)
            return labels
        
        print("train_rating_values:",train_rating_values)

        self.train_enc_graph = self._generate_enc_graph(train_rating_pairs, train_rating_values, add_support=True)
        self.train_dec_graph = self._generate_dec_graph(train_rating_pairs)
        self.train_labels = _make_labels(train_rating_values)
        self.train_truths = th.FloatTensor(train_rating_values).to(device)

        #self.valid_enc_graph = self.train_enc_graph
        #self.valid_dec_graph = self._generate_dec_graph(valid_rating_pairs)
        #self.valid_labels = _make_labels(valid_rating_values)
        #self.valid_truths = th.FloatTensor(valid_rating_values).to(device)

        self.test_enc_graph = self._generate_enc_graph(all_train_rating_pairs, all_train_rating_values, add_support=True)
        self.test_dec_graph = self._generate_dec_graph(test_rating_pairs)
        self.test_labels = _make_labels(test_rating_values)
        self.test_truths = th.FloatTensor(test_rating_values).to(device)

        #创建一个用来测试召回数据的图
        self.test_recall_labels = _make_labels(self.test_rating_info)
        
        #valid_recall_pair, valid_rating_matrix = self._generate_pair_value_for_recall(self.valid_rating_info)
        #self.valid_rating_matrix = th.FloatTensor(valid_rating_matrix).to(device)
        #self.valid_recall_dec_graph = self._generate_dec_graph(valid_recall_pair)


        #test_recall_pair, test_rating_matrix = self._generate_pair_value_for_recall_new(user_list, item_len)
        #test_recall_pair = self._generate_pair_value_for_recall_new(user_list, item_len)
        #self.test_rating_matrix = th.FloatTensor(test_rating_matrix).to(device)
        #self.test_recall_dec_graph = self._generate_dec_graph(test_recall_pair)


        def _npairs(graph):
            rst = 0
            for r in self.possible_rating_values:
                r = to_etype_name(r)
                rst += graph.number_of_edges(str(r))
            return rst

        print("Train enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.train_enc_graph.number_of_nodes('user'), self.train_enc_graph.number_of_nodes('movie'),
            _npairs(self.train_enc_graph)))
        print("Train dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.train_dec_graph.number_of_nodes('user'), self.train_dec_graph.number_of_nodes('movie'),
            self.train_dec_graph.number_of_edges()))
        # print("Valid enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
        #     self.valid_enc_graph.number_of_nodes('user'), self.valid_enc_graph.number_of_nodes('movie'),
        #     _npairs(self.valid_enc_graph)))
        # print("Valid dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
        #     self.valid_dec_graph.number_of_nodes('user'), self.valid_dec_graph.number_of_nodes('movie'),
        #     self.valid_dec_graph.number_of_edges()))
        # print("Test enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
        #     self.test_enc_graph.number_of_nodes('user'), self.test_enc_graph.number_of_nodes('movie'),
        #     _npairs(self.test_enc_graph)))
        # print("Test dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
        #     self.test_dec_graph.number_of_nodes('user'), self.test_dec_graph.number_of_nodes('movie'),
        #     self.test_dec_graph.number_of_edges()))

    def sample_negative(self, ratings, sample_rate, random_number):
        #"""return all negative items & 100 sampled negative items"""
        random.seed(random_number)
        interact_status = ratings.groupby('user_id')['movie_id'].apply(set).reset_index().rename(columns={'itemId': 'interacted_items'})
        #print(interact_status)
        #item_list = set(item_list)
        interact_status['negative_items'] = interact_status['movie_id'].apply(lambda x: self.item_poll - x)
        #print(interact_status['negative_items'])
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, sample_rate))
        return interact_status[['user_id', 'negative_items', 'negative_samples']]

    def negative_all(self, ratings):
        #"""return all negative items """
        interact_status = ratings.groupby('user_id')['movie_id'].apply(set).reset_index().rename(columns={'itemId': 'interacted_items'})
        #print(interact_status)
        #item_list = set(item_list)
        interact_status['negative_items'] = interact_status['movie_id'].apply(lambda x: list(self.item_poll - x))
        #print(interact_status['negative_items'])
        return interact_status[['user_id', 'negative_items']]

    def _generate_pair_value_for_zero(self,train_ratings, negatives):
        
        train_ratings = pd.merge(self.all_train_rating_info, negatives[['user_id', 'negative_samples']], on='user_id')
        train_users, train_items, negative_users, negative_items = [], [], [], []
        for row in train_ratings.itertuples():
            #train_users.append(int(row.userId))
            #train_items.append(int(row.itemId))
            for i in range(len(row.negative_samples)):
                negative_users.append(int(self.global_user_id_map[row.user_id]))
                negative_items.append(int(self.global_movie_id_map[row.negative_samples[i]]))
        rating_pairs = (np.array(negative_users) ,np.array(negative_items))
        rating_values = np.zeros(len(rating_pairs[0]))
        #print("rating_pairs[0].shape:", rating_pairs[0].shape)
        #print("rating_values:",rating_values.shape)
        assert len(rating_values) == len(rating_pairs[0])
        #print(rating_pairs, rating_values)
        return rating_pairs, rating_values

    def _generate_pair_value_for_recall(self, rating_info):
        """
        主要关注user
        """
        unique_user_list = list(pd.unique(rating_info["user_id"]))
        print("unique_user_list", len(unique_user_list))
        #print("unique_user_list", unique_user_list[0:10])

        unique_user_list = np.array([self.global_user_id_map[ele] for ele in unique_user_list])
        #print("unique_user_list",unique_user_list[0:10])
        user_length = len(unique_user_list)
        #rating_pairs1 = (np.array([[self.global_user_id_map[ele]] * self._num_movie for ele in user_list],
        #rating_pairs1 = (np.array([[self.global_user_id_map[ele]] * self._num_movie for ele in unique_user_list],
        rating_pairs1 = (np.array([[ele] * self._num_movie for ele in unique_user_list],
                                 dtype=np.int64).flatten(),
                        np.array(np.array([[np.arange(self._num_movie)] * user_length]).flatten(),
                                 dtype=np.int64))
                        #np.array([self.global_movie_id_map[ele] for ele in rating_info["movie_id"]],
                        #         dtype=np.int64))

        rating_pairs2 = (np.array([self.global_user_id_map[ele] for ele in rating_info["user_id"]],
                                 dtype=np.int64),
                        np.array([self.global_movie_id_map[ele] for ele in rating_info["movie_id"]],
                                 dtype=np.int64))

        #rating_values = user_list.astype(np.float32)
        rating_values = rating_info["rating"].values.astype(np.float32)
        rating_matrix = coo_matrix((rating_values, rating_pairs2), shape=(self._num_user, self._num_movie)).toarray()
        print("rating_matrix:",rating_matrix.shape)
        rating_matrix_cut = rating_matrix[unique_user_list]
        print("rating_matrix:",rating_matrix_cut.shape)
        return rating_pairs1, rating_matrix_cut

    def _generate_pair_value_for_recall_new(self, user_list, item_len):
        """
        主要关注user, 每128个user进行一次构建一个新的decode图
        输入：
        1. 当前测试的userlist
        2. 当前测试的item长度
        """
        unique_user_list = user_list
        print("unique_user_list", len(unique_user_list))
        #print("unique_user_list", unique_user_list[0:10])

        #unique_user_list = np.array([self.global_user_id_map[ele] for ele in unique_user_list])
        #print("unique_user_list",unique_user_list[0:10])
        user_length = len(unique_user_list)
        #rating_pairs1 = (np.array([[self.global_user_id_map[ele]] * self._num_movie for ele in user_list],
        #rating_pairs1 = (np.array([[self.global_user_id_map[ele]] * self._num_movie for ele in unique_user_list],
        rating_pairs1 = (np.array([[ele] * item_len for ele in unique_user_list],
                                 dtype=np.int64).flatten(),
                        np.array(np.array([[np.arange(item_len)] * user_length]).flatten(),
                                 dtype=np.int64))
                        #np.array([self.global_movie_id_map[ele] for ele in rating_info["movie_id"]],
                        #         dtype=np.int64))

        # rating_pairs2 = (np.array([self.global_user_id_map[ele] for ele in rating_info["user_id"]],
        #                          dtype=np.int64),
        #                 np.array([self.global_movie_id_map[ele] for ele in rating_info["movie_id"]],
        #                          dtype=np.int64))

        #rating_values = user_list.astype(np.float32)
        # rating_values = rating_info["rating"].values.astype(np.float32)
        # rating_matrix = coo_matrix((rating_values, rating_pairs2), shape=(self._num_user, self._num_movie)).toarray()
        # print("rating_matrix:",rating_matrix.shape)
        # rating_matrix_cut = rating_matrix[unique_user_list]
        # print("rating_matrix:",rating_matrix_cut.shape)
        return rating_pairs1#, rating_matrix_cut

    def _generate_pair_value(self, rating_info):
        rating_pairs = (np.array([self.global_user_id_map[ele] for ele in rating_info["user_id"]],
                                 dtype=np.int64),
                        np.array([self.global_movie_id_map[ele] for ele in rating_info["movie_id"]],
                                 dtype=np.int64))
        rating_values = rating_info["rating"].values.astype(np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False):
        user_movie_R = np.zeros((self._num_user, self._num_movie), dtype=np.float32)
        user_movie_R[rating_pairs] = rating_values

        data_dict = dict()
        num_nodes_dict = {'user': self._num_user, 'movie': self._num_movie}
        rating_row, rating_col = rating_pairs
        for rating in self.possible_rating_values:
            ridx = np.where(rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = to_etype_name(rating)
            data_dict.update({
                ('user', str(rating), 'movie'): (rrow, rcol),
                ('movie', 'rev-%s' % str(rating), 'user'): (rcol, rrow)
            })
        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        # sanity check
        #assert len(rating_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype('float32')
                x[x == 0.] = np.inf
                x = th.FloatTensor(1. / np.sqrt(x))
                return x.unsqueeze(1)
            user_ci = []
            user_cj = []
            movie_ci = []
            movie_cj = []
            for r in self.possible_rating_values:
                r = to_etype_name(r)
                user_ci.append(graph['rev-%s' % r].in_degrees())
                movie_ci.append(graph[r].in_degrees())
                if self._symm:
                    user_cj.append(graph[r].out_degrees())
                    movie_cj.append(graph['rev-%s' % r].out_degrees())
                else:
                    user_cj.append(th.zeros((self.num_user,)))
                    movie_cj.append(th.zeros((self.num_movie,)))
            user_ci = _calc_norm(sum(user_ci))
            movie_ci = _calc_norm(sum(movie_ci))
            if self._symm:
                user_cj = _calc_norm(sum(user_cj))
                movie_cj = _calc_norm(sum(movie_cj))
            else:
                user_cj = th.ones(self.num_user,)
                movie_cj = th.ones(self.num_movie,)
            graph.nodes['user'].data.update({'ci' : user_ci, 'cj' : user_cj})
            graph.nodes['movie'].data.update({'ci' : movie_ci, 'cj' : movie_cj})

        return graph

    def _generate_dec_graph(self, rating_pairs):
        ones = np.ones_like(rating_pairs[0])
        user_movie_ratings_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.num_user, self.num_movie), dtype=np.float32)
        g = dgl.bipartite_from_scipy(user_movie_ratings_coo, utype='_U', etype='_E', vtype='_V')
        return dgl.heterograph({('user', 'rate', 'movie'): g.edges()}, 
                               num_nodes_dict={'user': self.num_user, 'movie': self.num_movie})

    @property
    def num_links(self):
        return self.possible_rating_values.size

    @property
    def num_user(self):
        return self._num_user

    @property
    def num_movie(self):
        return self._num_movie

    def _drop_unseen_nodes(self, orign_info, cmp_col_name, reserved_ids_set, label):
        # print("  -----------------")
        # print("{}: {}(reserved) v.s. {}(from info)".format(label, len(reserved_ids_set),
        #                                                      len(set(orign_info[cmp_col_name].values))))
        if reserved_ids_set != set(orign_info[cmp_col_name].values):
            pd_rating_ids = pd.DataFrame(list(reserved_ids_set), columns=["id_graph"])
            # print("\torign_info: ({}, {})".format(orign_info.shape[0], orign_info.shape[1]))
            data_info = orign_info.merge(pd_rating_ids, left_on=cmp_col_name, right_on='id_graph', how='outer')
            data_info = data_info.dropna(subset=[cmp_col_name, 'id_graph'])
            data_info = data_info.drop(columns=["id_graph"])
            data_info = data_info.reset_index(drop=True)
            # print("\tAfter dropping, data shape: ({}, {})".format(data_info.shape[0], data_info.shape[1]))
            return data_info
        else:
            orign_info = orign_info.reset_index(drop=True)
            return orign_info

    def _load_raw_rates(self, file_path, sep):
        rating_info = pd.read_csv(
            file_path, sep=sep, header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            dtype={'user_id': np.int32, 'movie_id' : np.int32,
                   'ratings': np.float32, 'timestamp': np.int64}, engine='python',encoding="ISO-8859-1")
        return rating_info

    def _load_raw_rates_taobao(self, file_path, sep):
        rating_info = pd.read_csv(
            file_path, sep=sep, header=None,
            names=['user_id', 'movie_id', 'rating'],
            dtype={'user_id': np.int32, 'movie_id' : np.int32,
                   'ratings': np.int32}, engine='python',encoding="ISO-8859-1")
        return rating_info

    def _load_raw_rates_Tmall(self, file_path, sep):
        rating_info = pd.read_csv(
            file_path, sep=sep, header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            dtype={'user_id': np.int32, 'movie_id' : np.int32,
                   'ratings': np.float32, 'timestamp': np.int64}, engine='python',encoding="ISO-8859-1")
        return rating_info

    def _load_raw_user_info(self):
        if self._name == 'ml-100k':
            self.user_info = pd.read_csv(os.path.join(self._dir, 'u.user'), sep='|', header=None,
                                    names=['id', 'age', 'gender', 'occupation', 'zip_code'], engine='python',encoding="ISO-8859-1")
        elif self._name == 'ml-1m':
            self.user_info = pd.read_csv(os.path.join(self._dir, 'users.dat'), sep='::', header=None,
                                    names=['id', 'gender', 'age', 'occupation', 'zip_code'], engine='python',encoding="ISO-8859-1")
        elif self._name == 'ml-10m':
            rating_info = pd.read_csv(
                os.path.join(self._dir, 'ratings.dat'), sep='::', header=None,
                names=['user_id', 'movie_id', 'rating', 'timestamp'],
                dtype={'user_id': np.int32, 'movie_id': np.int32, 'ratings': np.float32,
                       'timestamp': np.int64}, engine='python',encoding="ISO-8859-1")
            self.user_info = pd.DataFrame(np.unique(rating_info['user_id'].values.astype(np.int32)),
                                     columns=['id'])
        else:
            raise NotImplementedError

    def _load_raw_movie_info(self):
        if self._name == 'ml-100k':
            GENRES = GENRES_ML_100K
        elif self._name == 'ml-1m':
            GENRES = GENRES_ML_1M
        elif self._name == 'ml-10m':
            GENRES = GENRES_ML_10M
        else:
            raise NotImplementedError

        if self._name == 'ml-100k':
            file_path = os.path.join(self._dir, 'u.item')
            self.movie_info = pd.read_csv(file_path, sep='|', header=None,
                                          names=['id', 'title', 'release_date', 'video_release_date', 'url'] + GENRES,
                                          engine='python',encoding="ISO-8859-1")
        elif self._name == 'ml-1m' or self._name == 'ml-10m':
            file_path = os.path.join(self._dir, 'movies.dat')
            movie_info = pd.read_csv(file_path, sep='::', header=None,
                                     names=['id', 'title', 'genres'], engine='python',encoding="ISO-8859-1")
            genre_map = {ele: i for i, ele in enumerate(GENRES)}
            genre_map['Children\'s'] = genre_map['Children']
            genre_map['Childrens'] = genre_map['Children']
            movie_genres = np.zeros(shape=(movie_info.shape[0], len(GENRES)), dtype=np.float32)
            for i, genres in enumerate(movie_info['genres']):
                for ele in genres.split('|'):
                    if ele in genre_map:
                        movie_genres[i, genre_map[ele]] = 1.0
                    else:
                        print('genres not found, filled with unknown: {}'.format(genres))
                        movie_genres[i, genre_map['unknown']] = 1.0
            for idx, genre_name in enumerate(GENRES):
                assert idx == genre_map[genre_name]
                movie_info[genre_name] = movie_genres[:, idx]
            self.movie_info = movie_info.drop(columns=["genres"])
        else:
            raise NotImplementedError

