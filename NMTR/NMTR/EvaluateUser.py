'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)
@author: hexiangnan
'''
import math
import heapq # for retrieval topK
from multiprocessing import cpu_count
from multiprocessing import Pool
import numpy as np
from time import time
#from numba import jit, autojit
# Global variables that are shared across processes
_model = None
_sess = None
_dataset = None
_K = None
_DictList = None
_gtItem = None
_user_prediction = None
_model_name = None
_feed_dict = None

def init_evaluate_model(model, dataset, args):
    DictList = []

    for idx in xrange(len(dataset.testRatings)):
        user, gtItem = dataset.testRatings[idx]
        items = range(dataset.num_items) # rank on all items
        items.append(gtItem)
        user_input = np.full(len(items), user, dtype='int32')[:, None]
        item_input = np.array(items)[:,None]
        if args.model == 'CMF':
            feed_dict = {model.user_input: user_input, model.item_input_buy: item_input}
        else:
            feed_dict = {model.user_input: user_input,  model.item_input: item_input}
        DictList.append(feed_dict)

    # print("already initiate the evaluate model...")
    return DictList


def gen_feed_dict(dataset):
    DictList = []
    for idx in xrange(len(dataset.testRatings)):
        user, gtItem = dataset.testRatings[idx]
        items = range(dataset.num_items) # rank on all items
        items.append(gtItem)
        user_input = np.full(len(items), user, dtype='int32')[:, None]
        item_input = np.array(items)[:,None]
        feed_dict = {'input_data/user_input:0': user_input, 
            'input_data/item_input:0': item_input}
        DictList.append(feed_dict)
    return DictList


def eval(model, sess, dataset, DictList, args, behave_type = None):
    global _model
    global _K
    global _DictList
    global _sess
    global _dataset
    global _gtItem
    global _user_prediction
    global _model_name
    global _feed_dict

    _dataset = dataset
    _model = model
    _sess = sess
    _K = args.topK
    _model_name = args.model


    if (args.dataset == 'ali2') and (behave_type == 'cart'):
        behave_type = 'buy'

    hits50,hits100,hits200, ndcgs50,ndcgs100,ndcgs200, ranks, _gtItem, _user_prediction = [], [], [], [], [],[],[],[],[]

    # give predictions on users
    # for idx in xrange(len(_DictList)):
    #     if args.model == 'Multi_GMF':
    #         _gtItem.append(_dataset[0].testRatings[idx][1])
    #         _user_prediction.append(_sess.run(_model.score_buy, feed_dict = _DictList[idx]))

    #     else:
    #         _gtItem.append(_dataset.testRatings[idx][1])
    #         _user_prediction.append(_sess.run(_model.output, feed_dict = _DictList[idx]))

    # cpu_num = 4
    # pool = Pool(cpu_num)
    # res = pool.map(_eval_one_rating, range(len(_DictList)))
    # pool.close()
    # pool.join()
    # hits = [r[0] for r in res]
    # ndcgs = [r[1] for r in res]

    _DictList = DictList
    for idx in xrange(len(_DictList)):
        (hr50,hr100,hr200, ndcg50,ndcg100,ndcg200, rank) = _eval_one_rating(idx, behave_type)
        hits50.append(hr50)
        ndcgs50.append(ndcg50)

        hits100.append(hr100)
        ndcgs100.append(ndcg100)

        hits200.append(hr200)
        ndcgs200.append(ndcg200)
        ranks.append(rank)

    return (hits50,hits100,hits200, ndcgs50,ndcgs100,ndcgs200, ranks)



def eval_FISM(model, sess, dataset, args, behave_type = None):
    global _model
    global _K
    global _DictList
    global _sess
    global _dataset
    global _gtItem
    global _user_prediction
    global _model_name
    global _feed_dict

    _dataset = dataset
    _model = model
    _sess = sess
    _K = args.topK
    _model_name = args.model

    hits, ndcgs, ranks, _gtItem, _user_prediction = [], [], [], [], []

    for idx in xrange(len(dataset.testRatings)):
        t1 = time()
        user, gtItem = dataset.testRatings[idx]
        items = range(dataset.num_items) # rank on all items
        items.append(gtItem)
        user_input = np.full(len(items), user, dtype='int32')[:, None]
        item_input = np.array(items)[:,None]
        # item rate / item_num 
        item_rate, item_num = [], []
        item_rate_1 = dataset.trainDict[user]['buy']
        for i in items: 
            item_rate_2 = filter(lambda x:x != i, item_rate_1) 
            item_num.append(len(item_rate_2))
            item_rate_2 = item_rate_2 + [dataset.num_items]*(dataset.max_rate - len(item_rate_2))
            item_rate.append(item_rate_2)
            assert len(item_rate_2) == dataset.max_rate
        feed_dict = {model.user_input: user_input,  model.item_input: item_input,
                     model.item_rate: np.array(item_rate), model.item_num: np.array(item_num).reshape(-1, 1)}
        _feed_dict = feed_dict
        (hr, ndcg, rank) = _eval_one_rating_FISM(idx)
        hits.append(hr)
        ndcgs.append(ndcg)
        ranks.append(rank)

    return (hits, ndcgs, ranks)








def _eval_one_rating_FISM(idx):
    gtItem = _dataset.testRatings[idx][1]
    predictions = _sess.run(_model.output, feed_dict = _feed_dict)

    rank = 0
    rank_score = predictions[gtItem]

    for i in predictions:
        if i > rank_score:
            rank += 1

    # deal with anomoly
    if (predictions[gtItem] == predictions[0]) and (predictions[gtItem] == predictions[1]) and (predictions[gtItem] == predictions[2]):
        rank = 99999
    
    if rank < _K:
        hr = 1
        ndcg = math.log(2) / math.log(rank + 2)
    else:
        hr = 0
        ndcg = 0

    # real ranking should be this
    rank_real = rank + 1
    
    return (hr, ndcg, rank_real)





def _eval_one_rating(idx, behave_type):

    # predictions = _user_prediction[idx]
    # gtItem = _gtItem[idx]

    if _model_name in ['Multi_GMF', 'Multi_MLP', 'Multi_NCF']:
        gtItem = _dataset[0].testRatings[idx][1]
        Train_k= _dataset[0].trainMatrix
        if behave_type == 'ipv':
            predictions = _sess.run(_model.score_ipv, feed_dict = _DictList[idx])
        elif behave_type == 'cart':
            predictions = _sess.run(_model.score_cart, feed_dict = _DictList[idx])
        else:
            predictions = _sess.run(_model.score_buy, feed_dict = _DictList[idx])

    elif _model_name == 'CMF':
        gtItem = _dataset.testRatings[idx][1]
        Train_k= _dataset.trainDict[idx]['buy']

        predictions = _sess.run(_model.output_buy, feed_dict = _DictList[idx])
    
    else:
        gtItem = _dataset.testRatings[idx][1]
        Train_k= _dataset.trainMatrix

        if behave_type == 'ipv':
            predictions = _sess.run(_model.score_ipv, feed_dict = _DictList[idx])
        elif behave_type == 'cart':
            predictions = _sess.run(_model.score_cart, feed_dict = _DictList[idx])
        elif behave_type == 'buy':
            predictions = _sess.run(_model.score_buy, feed_dict = _DictList[idx])
        else:
            predictions = _sess.run(_model.output, feed_dict = _DictList[idx])

    rank = 0
    rank_score = predictions[gtItem]
    tk=0
    #print Train_k.keys()

    if _model_name == 'CMF':
        for i in predictions:
            if i > rank_score:
                if tk not in Train_k:
                    rank += 1

            tk=tk+1
    else:
        for i in predictions:
            if i > rank_score:
                if not Train_k.has_key((idx,tk)):
                    rank += 1
            tk=tk+1

    # deal with anomoly
    if (predictions[gtItem] == predictions[0]) and (predictions[gtItem] == predictions[1]) and (predictions[gtItem] == predictions[2]):
        rank = 99999
    

    if rank < 50:#+Train_k:
        hr50 = 1
        ndcg50 = math.log(2) / math.log(rank + 2)
    else:
        hr50 = 0
        ndcg50 = 0

    if rank < 100:#+Train_k:
        hr100 = 1
        ndcg100 = math.log(2) / math.log(rank + 2)
    else:
        hr100 = 0
        ndcg100 = 0

    if rank < 200:#+Train_k:
        hr200 = 1
        ndcg200 = math.log(2) / math.log(rank + 2)
    else:
        hr200 = 0
        ndcg200 = 0

    # real ranking should be this
    rank_real = rank + 1
    
    return (hr50,hr100,hr200, ndcg50,ndcg100,ndcg200, rank_real)



