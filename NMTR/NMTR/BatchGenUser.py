import numpy as np
import pandas as pd
from multiprocessing import Pool
from multiprocessing import cpu_count
from time import time 
import random

_user_input = None
_item_input = None
_item_input_neg = None
_item_rate =None
_item_num = None
_labels = None
_labels_3 = None
_batch_size = None
_index = None
_dataset_ipv = None
_dataset_cart = None
_dataset_buy = None 
_dataset = None

# input: dataset(Mat, List, Rating, Negatives), batch_choice, num_negatives
# output: [_user_input_list, _item_input_list, _labels_list]
def sampling(args, dataset, num_negatives):

    if args.model == "FISM":
        _user_input, _item_input, _item_rate, _item_num, _labels = [], [], [], [], []
        num_users, num_items = dataset.num_users, dataset.num_items
        for u in dataset.trainDict.keys():
            item_rate = dataset.trainDict[u]['buy']
            item_num = len(item_rate)   # all rated items of user u
            for i in item_rate:
                item_rate_2 = filter(lambda x:x != i, item_rate)
                _user_input.append(u)
                _item_input.append(i)
                _item_rate.append(item_rate_2)
                _item_num.append(item_num-1)
                _labels.append(1)
                # negative instances
                for t in xrange(num_negatives):
                    j = np.random.randint(num_items)
                    while j in item_rate:
                        j = np.random.randint(num_items)
                    _user_input.append(u)
                    _item_input.append(j)
                    _item_rate.append(item_rate)
                    _item_num.append(item_num)
                    _labels.append(0)

        # additional manipulation on _item_rate
        max_rate = max(map(lambda x: len(x), _item_rate))
        _item_rate_fixed = []
        for i in _item_rate:
            _item_rate_fixed.append(i + [num_items]*(max_rate - len(i)))

        return _user_input, _item_input, _item_rate_fixed, _item_num, _labels


    else:
        _user_input, _item_input, _labels = [], [], []
        num_users, num_items =  dataset.num_users, dataset.num_items

        if args.en_MC == 'yes':
        # enable multi-channel sampling for single behavior models 
            for u in dataset.trainDict.keys():
                log = dataset.trainDict[u]
                for i in log['buy']:
                    _user_input += [u] * (num_negatives+1)
                    _item_input += [i]
                    _labels +=  [1]
                    uo_neg_prob = args.beta
                    cart_weight, ipv_weight = len(log['cart']) / 2.0, len(log['ipv']) / 4.0
                    # ipv_neg_prob = (1 - args.beta) * ipv_weight / (cart_weight + ipv_weight)
                    ipv_neg_prob = 1-uo_neg_prob
                    _item_input += _sample_neg_accord_prob([uo_neg_prob, ipv_neg_prob], log, num_negatives, num_items)
                    _labels += [0] * (num_negatives)
         
        else:
            for (u, i) in dataset.trainMatrix.keys():
                # positive instance
                _user_input.append(u)
                _item_input.append(i)
                _labels.append(1)
                # negative instances
                for t in xrange(num_negatives):
                    j = np.random.randint(num_items)
                    while dataset.trainMatrix.has_key((u, j)):
                        j = np.random.randint(num_items)
                    _user_input.append(u)
                    _item_input.append(j)
                    _labels.append(0)
        assert (len(_user_input) == len(_item_input)) and (len(_user_input) == len(_labels))
        
        return _user_input, _item_input, _labels


def shuffle(samples, batch_size, args):
    global _user_input
    global _item_input
    global _item_rate
    global _item_num  
    global _labels
    global _batch_size
    global _index

    if args.model == 'FISM':
        _user_input, _item_input, _item_rate, _item_num, _labels = samples
        _batch_size = batch_size
        _index = range(len(_user_input))
        np.random.shuffle(_index)
        num_batch = len(_user_input) // _batch_size
        pool = Pool(4)

        print ('num_batch:%d, all_entries:%d, batch_size:%d, labels:%d' %(
                num_batch, len(_user_input), _batch_size, len(_labels)))
        t1 = time()
        res = pool.map(_get_train_batch_FISM, range(num_batch))
        pool.close()
        pool.join()

        user_list = [r[0] for r in res]
        item_list = [r[1] for r in res]
        item_rate_list = [r[2] for r in res]
        item_num_list = [r[3] for r in res]
        labels_list = [r[4] for r in res]
        
        # for i in range(num_batch):
        #     user_list = _get_train_batch_FISM(i)[0]
        #     item_list = _get_train_batch_FISM(i)[1]
        #     item_rate_list = _get_train_batch_FISM(i)[2]
        #     item_num_list = _get_train_batch_FISM(i)[3]
        #     labels_list = _get_train_batch_FISM(i)[4]      

        print('shuffle time: %d' %(time() - t1))
        return user_list, item_list, item_rate_list, item_num_list, labels_list

    elif args.model in ['pure_GMF', 'pure_MLP', 'pure_NCF']:
        _user_input, _item_input, _labels = samples
        _batch_size = batch_size
        _index = range(len(_labels))
        np.random.shuffle(_index)
        num_batch = len(_labels) // _batch_size
        pool = Pool(4)
        res = pool.map(_get_train_batch, range(num_batch))
        pool.close()
        pool.join()

        user_list = [r[0] for r in res]
        item_list = [r[1] for r in res]
        labels_list = [r[2] for r in res]

        return user_list, item_list, labels_list

    else:
        _user_input, _item_input, _item_input_neg = samples
        _batch_size = batch_size
        _index = range(len(_user_input))
        np.random.shuffle(_index)
        num_batch = len(_user_input) // _batch_size
        pool = Pool(4)
        res = pool.map(_get_train_batch, range(num_batch))
        pool.close()
        pool.join()

        user_list = [r[0] for r in res]
        item_list = [r[1] for r in res]
        labels_list = [r[2] for r in res]
        return user_list, item_list, labels_list

def _get_train_batch(i):
    user_batch, item_batch, labels_batch = [], [], []
    begin = i * _batch_size
    for idx in range(begin, begin + _batch_size):
        user_batch.append(_user_input[_index[idx]])
        item_batch.append(_item_input[_index[idx]])
        labels_batch.append(_labels[_index[idx]])
    return np.array(user_batch), np.array(item_batch), np.array(labels_batch)


def _get_train_batch_FISM(i):
    user_batch, item_batch, item_rate_batch, item_num_batch, labels_batch = [], [], [], [], []
    begin = i * _batch_size
    for idx in range(begin, begin + _batch_size):
        user_batch.append(_user_input[_index[idx]])
        item_batch.append(_item_input[_index[idx]])
        item_rate_batch.append(_item_rate[_index[idx]])
        item_num_batch.append(_item_num[_index[idx]])       
        labels_batch.append(_labels[_index[idx]])
    return np.array(user_batch), np.array(item_batch), np.array(item_rate_batch), np.array(item_num_batch), np.array(labels_batch)


def _sample_neg_accord_prob(prob, log, num_negatives, num_items):
    
    [uo_neg_prob, ipv_neg_prob] = prob
    neg = []
    for _ in range(num_negatives):
        j = random.random()
        if j < uo_neg_prob:
            neg += _sample_unobserved(1, num_items, log)
        # elif j < uo_neg_prob + ipv_neg_prob:
        #     tem = random.sample(log['ipv'], 1)[0]
        #     while (tem in neg) or (tem in log['buy']) or (tem in log['cart']):
        #         tem = random.sample(log['ipv'], 1)[0]
        #     neg.append(tem)
        #     # neg += random.sample(log['ipv'], 1)
        # else:
        #     tem = random.sample(log['cart'], 1)[0]
        #     while (tem in neg) or (tem in log['buy']):
        #         tem = random.sample(log['cart'], 1)[0]
        #     neg.append(tem)
        else:
            tem = random.sample(log['ipv'], 1)[0]
            while tem in log['buy']:
                tem = random.sample(log['ipv'], 1)[0]
            neg.append(tem)
    return neg


# dataset [dataset_ipv, dataset_cart, dataset_buy]
def sampling_3(args, dataset, num_negatives):
    start_time = time()
    _user_input, _item_input, _item_input_neg, _labels = [], [], [], []

    if args.model == "Multi_BPR":
        num_users, num_items = dataset.num_users, dataset.num_items
        if args.neg_sample_tech == 'fix':       
            for u in dataset.trainDict.keys():
                log = dataset.trainDict[u]
                for i in log['buy']:
                    neg_cart_num = min(num_negatives, len(log['cart']))
                    neg_ipv_num = min(int(num_negatives / 2), len(log['ipv']))
                    neg_uo_num = int(num_negatives / 4)
                    _user_input += [u] * (neg_cart_num + neg_ipv_num + neg_uo_num)
                    _item_input += [i] * (neg_cart_num + neg_ipv_num + neg_uo_num)
                    _item_input_neg += random.sample(log['cart'], neg_cart_num)
                    _item_input_neg += random.sample(log['ipv'], neg_ipv_num)
                    _item_input_neg += _sample_unobserved(neg_uo_num, num_items, log)

                for i in log['cart']:
                    neg_ipv_num = min(num_negatives, len(log['ipv']))
                    neg_uo_num = int(num_negatives / 2)
                    _user_input += [u] * (neg_ipv_num + neg_uo_num)
                    _item_input += [i] * (neg_ipv_num + neg_uo_num)
                    _item_input_neg += random.sample(log['ipv'], neg_ipv_num)
                    _item_input_neg += _sample_unobserved(neg_uo_num, num_items, log)

                for i in log['ipv']:
                    neg_uo_num = int(num_negatives)
                    _user_input += [u] * neg_uo_num
                    _item_input += [i] * neg_uo_num
                    _item_input_neg += _sample_unobserved(neg_uo_num, num_items, log)

        elif args.neg_sample_tech == 'prob':
            for u in dataset.trainDict.keys():
                log = dataset.trainDict[u]
                for i in log['buy']:
                    _user_input += [u] * num_negatives
                    _item_input += [i] * num_negatives
                    uo_neg_prob = args.beta
                    cart_weight, ipv_weight = len(log['cart']) / 2.0, len(log['ipv']) / 4.0
                    ipv_neg_prob = (1 - args.beta) * ipv_weight / (cart_weight + ipv_weight)
                    _item_input_neg += _sample_neg_accord_prob([uo_neg_prob, ipv_neg_prob], log, num_negatives, num_items)
            
                for i in log['cart']:
                    _user_input += [u] * num_negatives
                    _item_input += [i] * num_negatives
                    uo_neg_prob = args.beta
                    ipv_neg_prob = 1 - args.beta
                    _item_input_neg += _sample_neg_accord_prob([uo_neg_prob, ipv_neg_prob], log, num_negatives, num_items)

                for i in log['ipv']:
                    _user_input += [u] * num_negatives
                    _item_input += [i] * num_negatives
                    _item_input_neg += _sample_neg_accord_prob([1, 0], log, num_negatives, num_items)

        return _user_input, _item_input, _item_input_neg
    
    elif args.model == "BPR":
        num_users, num_items = dataset.num_users, dataset.num_items
        for u in dataset.trainDict.keys():
            log = dataset.trainDict[u]['buy'] + dataset.trainDict[u]['cart'] + dataset.trainDict[u]['ipv']
            _user_input += [u] * num_negatives * len(log)
            for i in log:
                _item_input += [i] * num_negatives
                for _ in range(num_negatives):
                    j = np.random.randint(num_items)
                    while j in log:
                        j = np.random.randint(num_items)
                    _item_input_neg.append(j)
        # print(len(_user_input), len(_item_input), len(_item_input_neg))
        return _user_input, _item_input, _item_input_neg
    
    elif args.model == 'CMF':
        num_users, num_items = dataset.num_users, dataset.num_items
        _item_input_buy, _item_input_cart, _item_input_ipv = [], [], []
        _labels_buy, _labels_cart, _labels_ipv = [], [], []
        
        for u in dataset.trainDict.keys():
            log_buy = dataset.trainDict[u]['buy']
            log_cart = dataset.trainDict[u]['cart']
            log_ipv = dataset.trainDict[u]['ipv']
            log = log_buy + log_cart + log_ipv

            tmp_buy, tmp_cart, tmp_ipv = [], [], []
            lmp_buy, lmp_cart, lmp_ipv = [], [], []

            for i in log:
                if i in log_buy:
                    tmp_buy += [i]
                    lmp_buy += [1]
                    for _ in range(num_negatives):
                        j = np.random.randint(num_items)
                        while j in log:
                            j = np.random.randint(num_items)
                        tmp_buy += [j]
                        lmp_buy += [0]
                elif i in log_cart:
                    tmp_cart += [i]
                    lmp_cart += [1]
                    for _ in range(num_negatives):
                        j = np.random.randint(num_items)
                        while j in log:
                            j = np.random.randint(num_items)
                        tmp_cart += [j]
                        lmp_cart += [0]
                else:
                    tmp_ipv += [i]
                    lmp_ipv += [1]
                    for _ in range(num_negatives):
                        j = np.random.randint(num_items)
                        while j in log:
                            j = np.random.randint(num_items)
                        tmp_ipv += [j]
                        lmp_ipv += [0]

            num_buy, num_cart, num_ipv = len(tmp_buy), len(tmp_cart), len(tmp_ipv)
            max_num = max(num_buy, num_cart, num_ipv)
            if num_buy == 0:
                num_buy = num_negatives
                for _ in range(num_negatives):
                    j = np.random.randint(num_items)
                    while j in log:
                        j = np.random.randint(num_items)
                    tmp_buy += [j]
                    lmp_buy += [0]
            if num_cart == 0:
                num_cart = num_negatives
                for _ in range(num_negatives):
                    j = np.random.randint(num_items)
                    while j in log:
                        j = np.random.randint(num_items)
                    tmp_cart += [j]
                    lmp_cart += [0]
            if num_ipv == 0:
                num_ipv = num_negatives
                for _ in range(num_negatives):
                    j = np.random.randint(num_items)
                    while j in log:
                        j = np.random.randint(num_items)
                    tmp_ipv += [j]
                    lmp_ipv += [0]

            _user_input += [u] * max_num
            for _ in range(max_num - num_buy):
                j = np.random.randint(num_buy)
                tmp_buy.append(tmp_buy[j])
                lmp_buy.append(lmp_buy[j])
            for _ in range(max_num - num_cart):
                j = np.random.randint(num_cart)
                tmp_cart.append(tmp_cart[j])
                lmp_cart.append(lmp_cart[j])
            for _ in range(max_num - num_ipv):
                j = np.random.randint(num_ipv)
                tmp_ipv.append(tmp_ipv[j])
                lmp_ipv.append(lmp_ipv[j])
            
            _item_input_buy += tmp_buy
            _item_input_cart += tmp_cart
            _item_input_ipv += tmp_ipv
            _labels_buy += lmp_buy
            _labels_cart += lmp_cart
            _labels_ipv += lmp_ipv

        return (_user_input, _item_input_buy, _item_input_cart, _item_input_ipv,
            _labels_buy, _labels_cart, _labels_ipv)


    else:
        # for printing buying loss only 
        _user_input_b, _item_input_b, _labels_b, _labels_v, _labels_c = [], [], [], [], []
        # load data and create matrix
        _dataset_ipv, _dataset_cart, _dataset_buy = dataset
        num_users, num_items =  _dataset_ipv.trainMatrix.shape
        if args.b_num == 3:
            _dataset = _dataset_ipv.trainMatrix + _dataset_cart.trainMatrix + _dataset_buy.trainMatrix # combining 3 behaviors in one matrix
        else:
            if args.b_2_type == 'vc':
                _dataset = _dataset_ipv.trainMatrix + _dataset_cart.trainMatrix
            elif args.b_2_type == 'cb':
                _dataset = _dataset_cart.trainMatrix + _dataset_buy.trainMatrix  
            else:
                _dataset = _dataset_ipv.trainMatrix + _dataset_buy.trainMatrix

        print('num_users: %d \nnum_items: %d \nnum_ipv: %d \nnum_cart: %d \nnum_buy: %d \nnum_all %d' 
            %(num_users,  num_items, _dataset_ipv.trainMatrix.nnz, _dataset_cart.trainMatrix.nnz, _dataset_buy.trainMatrix.nnz, _dataset.nnz))

        # define mapping for 1-d label to 3-d label
        label_map = {0:[0.0,0.0,0.0], 1:[0.0,0.0,1.0], 2:[0.0,1.0,1.0], 3:[1.0,1.0,1.0]}

        for (u, i) in _dataset.keys():
            # positive instance
            _user_input.append(u)
            _item_input.append(i)
            _labels.append(int(_dataset[u, i]))

            # buying behavior only
            if args.buy_loss == 'yes':
                if _dataset[u, i] == 3:
                    _user_input_b.append(u)
                    _item_input_b.append(i)
                    _labels_b.append(1)
                    _labels_v.append(1)
                    _labels_c.append(1)
                else:
                    pass

            # negative instances
            for t in xrange(num_negatives):
                j = np.random.randint(num_items)
                while _dataset.has_key((u, j)):
                    j = np.random.randint(num_items)
                _user_input.append(u)
                _item_input.append(j)
                _labels.append(0)

                # buying behavior only
                if args.buy_loss == 'yes':
                    if _dataset[u, i] == 3:
                        _user_input_b.append(u)
                        _item_input_b.append(j)
                        _labels_b.append(0)
                        _labels_v.append(0)
                        _labels_c.append(0)                        
                    else:
                        pass

        _labels_sr = pd.Series(_labels)
        _labels_3 = _labels_sr.map(label_map)
        _labels_3 = _labels_3.tolist()
        
        print('len label is %d' % len(_user_input_b))

        if args.buy_loss == 'no':
            return _user_input, _item_input, _labels, _labels_3
        else:
            return _user_input, _item_input, _labels, _labels_3, _user_input_b, _item_input_b, _labels_v, _labels_c, _labels_b


def unshuffle(sample, batch_size):
    sample_num = len(sample)
    assert sample_num == 5
    batch_num = len(sample[0]) // batch_size

    sample_batch=[]
    for i in range(sample_num):
        k = 0
        sample_batch.append([])
        for j in range(batch_num):        
            sample_batch[i].append(sample[i][k:k+batch_size])
            k += batch_size

    return sample_batch



def _sample_unobserved(neg_uo_num, num_items, log):
    neg_uo = []
    for _ in range(neg_uo_num):
        j = np.random.randint(num_items)
        while j in log['buy'] or j in log['cart'] or j in log['ipv']:
            j = np.random.randint(num_items)
        neg_uo.append(j)
    return neg_uo

def shuffle_3(samples, batch_size, args):
    global _user_input
    global _item_input
    global _item_input_buy, _item_input_cart, _item_input_ipv
    global _labels_buy, _labels_cart, _labels_ipv
    global _item_input_neg
    global _labels
    global _labels_3
    global _batch_size
    global _index
    
    if args.model in ['Multi_GMF', 'Multi_MLP', 'Multi_NCF']:
        _user_input, _item_input, _labels, _labels_3 = samples
        _batch_size = batch_size
        _index = range(len(_labels))
        np.random.shuffle(_index)
        num_batch = len(_labels) // _batch_size
        pool = Pool(4)       
        res = pool.map(_get_train_batch_3, range(num_batch))
        pool.close()
        pool.join()
        user_list = [r[0] for r in res]
        item_list = [r[1] for r in res]
        argv_list = [r[2] for r in res]
        return user_list, item_list, argv_list

    elif 'BPR' in args.model:
        _user_input, _item_input, _item_input_neg = samples
        _batch_size = batch_size
        _index = range(len(_user_input))
        np.random.shuffle(_index)
        num_batch = len(_user_input) // _batch_size
        pool = Pool(4)
        res = pool.map(_get_train_batch_BPR, range(num_batch))
        pool.close()
        pool.join()
        user_list = [r[0] for r in res]
        item_list = [r[1] for r in res]
        argv_list = [r[2] for r in res]
        return user_list, item_list, argv_list
    
    elif 'CMF' in args.model:
        (_user_input, _item_input_buy, _item_input_cart, _item_input_ipv,
            _labels_buy, _labels_cart, _labels_ipv) = samples
        _batch_size = batch_size
        _index = range(len(_user_input))
        np.random.shuffle(_index)
        num_batch = len(_user_input) // _batch_size
        pool = Pool(4)
        res = pool.map(_get_train_batch_CMF, range(num_batch))
        pool.close()
        pool.join()
        user_list = [r[0] for r in res]
        item_buy_list = [r[1] for r in res]
        item_cart_list = [r[2] for r in res]
        item_ipv_list = [r[3] for r in res]
        labels_buy_list = [r[4] for r in res]
        labels_cart_list = [r[5] for r in res]
        labels_ipv_list = [r[6] for r in res]
        return (user_list, item_buy_list, item_cart_list, item_ipv_list,
            labels_buy_list, labels_cart_list, labels_ipv_list)
      

def _get_train_batch_3(i):
    user_batch, item_batch, labels_batch = [], [], []
    begin = i * _batch_size
    for idx in range(begin, begin + _batch_size):
        user_batch.append(_user_input[_index[idx]])
        item_batch.append(_item_input[_index[idx]])
        labels_batch.append(_labels_3[_index[idx]])
    return np.array(user_batch), np.array(item_batch), np.array(labels_batch)

def _get_train_batch_BPR(i):
    user_batch, item_batch, item_neg_batch = [], [], []
    begin = i * _batch_size
    for idx in range(begin, begin + _batch_size):
        user_batch.append(_user_input[_index[idx]])
        item_batch.append(_item_input[_index[idx]])
        item_neg_batch.append(_item_input_neg[_index[idx]])
    return np.array(user_batch), np.array(item_batch), np.array(item_neg_batch)

def _get_train_batch_CMF(i):
    (user_batch, item_buy_batch, item_cart_batch, item_ipv_batch,
        labels_buy_batch, labels_cart_batch, labels_ipv_batch) = [], [], [], [], [], [], []
    begin = i * _batch_size
    for idx in range(begin, begin + _batch_size):
        user_batch.append(_user_input[_index[idx]])
        item_buy_batch.append(_item_input_buy[_index[idx]])
        item_cart_batch.append(_item_input_cart[_index[idx]])
        item_ipv_batch.append(_item_input_ipv[_index[idx]])
        labels_buy_batch.append(_labels_buy[_index[idx]])
        labels_cart_batch.append(_labels_cart[_index[idx]])
        labels_ipv_batch.append(_labels_ipv[_index[idx]])

    return (np.array(user_batch), np.array(item_buy_batch), np.array(item_cart_batch),
        np.array(item_ipv_batch), np.array(labels_buy_batch), np.array(labels_cart_batch),
        np.array(labels_ipv_batch))










# def FISM_shift_data

