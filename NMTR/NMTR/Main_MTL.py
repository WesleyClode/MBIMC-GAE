from __future__ import division

import os
# os.chdir('/home/stu/gandahua/MBR/')
#os.chdir('/home/chenchong/Multi_Behavior/')

import numpy as np
import math

import logging

from time import time, sleep
from time import strftime
from time import localtime

from Models import FISM
from Models import MLP
from Models import GMF_controlled
from Models import GMF_Model
from Models import pure_GMF
from Models import GMF_FC
from Models import pure_NCF, pure_NCF_2
from Models import NCF_FC
from Models import Multi_GMF, Multi_MLP, Multi_NCF, Multi_NCF_2
from Models import BPR
from Models import pure_MLP
from Models import CMF

import BatchGenItem as BatchItem
import BatchGenUser as BatchUser

import EvaluateItem as EvalItem
import EvaluateUser as EvalUser

from Dataset import Dataset
# from Dataset import Dataset_cold
# from saver import GMFSaver
# from loader import GMFLoader

import argparse
import pickle as pkl

#import setproctitle

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from multiprocessing import Process, cpu_count, Semaphore, Lock, Pool, Queue, JoinableQueue


def parse_args():
    parser = argparse.ArgumentParser(description="Run RSGAN.")
    parser.add_argument('--dataset', nargs='?', default='ali',
                        help='Choose a dataset: bb1, bb2, bb3, ali, kaggle')
    parser.add_argument('--model', nargs='?', default='pure_NCF',
                        help='Choose model: GMF, MLP, FISM, Multi_GMF, Multi_BPR, BPR')
    parser.add_argument('--loss_func', nargs='?', default='logloss',
                        help='Choose loss: logloss, BPR')
    parser.add_argument('--verbose', type=int, default=10,
                        help='Interval of evaluation.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=2048,
                        help='batch_size')   
    parser.add_argument('--batch_size_ipv', nargs='?', type=int, default=2048,
                        help='batch_size_ipv')
    parser.add_argument('--batch_size_cart', nargs='?', type=int, default=512,
                        help='batch_size_cart')
    parser.add_argument('--batch_size_buy', nargs='?', type=int, default=256,
                        help='batch_size_buy')                         
    parser.add_argument('--batch_choice', nargs='?', default='user',
                        help='user: generate batches by user, fixed:batch_size: generate batches by batch size')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layer_num', type = int, nargs='?', default=3,
                        help='layer number')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--alpha', type=float, default=0,
                        help='Index of coefficient of FISM')
    parser.add_argument('--train_loss', type=bool, default=True,
                        help='Caculate training loss or not')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--process_name', nargs='?', default='MBR_default@gaochen',
                        help='Input process name.')
    parser.add_argument('--gpu', type=int, default=5,
                        help='GPU.')
    parser.add_argument('--evaluate', nargs='?', default='yes',
                        help='Evaluate or not.')
    parser.add_argument('--frozen', nargs='?', default='',
                        help='Frozen user or item')
    parser.add_argument('--optimizer', nargs='?', default='Adagrad',
                        help='Choose an optimizer')
    parser.add_argument('--add_fc', nargs='?', type=bool, default=True,
                        help='Add fully connected layer or not.')
    parser.add_argument('--plot_network', nargs='?', type=bool, default=False,
                        help='If choosing to plot network, the train will be skipped')
    parser.add_argument('--training_type', nargs='?', default='independent',
                        help='Choose type of training: independent or cascade')
    parser.add_argument('--pretrain', nargs='?', default='',
                        help='Load pre-trained vectors')
    parser.add_argument('--validate', nargs='?', type=bool, default=False,
                        help='Enable the calculation of validation losss during training')
    parser.add_argument('--batch_sample', nargs='?', type=bool, default=False,
                        help='generate batch samples for dataset')    
    parser.add_argument('--data_gen', nargs='?', type=bool, default=False,
                        help='generate dataset or not')       
    parser.add_argument('--topK', nargs='?', type=int, default=100,
                        help='topK for hr/ndcg')     
    parser.add_argument('--frozen_type', nargs='?', type=int, default=0,
                        help='0:no_frozen, 1:item_frozen, 2:all_frozen')    
    parser.add_argument('--loss_coefficient', nargs='?', default='[1/3,1/3,1/3]',
                        help='loss coefficient for Multi_GMF')     
    parser.add_argument('--multiprocess', nargs='?', default='no',
                        help='Evaluate multiprocessingly or not')     
    parser.add_argument('--trial_id', nargs='?', default='1',
                        help='Indicate trail id with same condition')    
    parser.add_argument('--recover', nargs='?', default='no',
                        help='recover result from the server')
    parser.add_argument('--beta', type=float, default=0.2,
                        help='Multi_BPR beta')
    parser.add_argument('--neg_sample_tech', nargs='?', default='prob',
                        help='Multi_BPR sample technique')    
    parser.add_argument('--en_MC', nargs='?', default='no',
                        help='enable Multi-Channel for single behavior model')
    parser.add_argument('--dropout', type=float, default=0.8,
                        help='dropout keep_prob')                                                                       
    parser.add_argument('--b_num', nargs='?', type=int, default=3,
                        help='control the behavior number in multitask learning')     
    parser.add_argument('--b_2_type', nargs='?', default='vb',
                        help='when b_num=2, three condition: vc->view/cart, cb->cart/buy, vb->view/buy')                              
    parser.add_argument('--half_behave', nargs='?', default='no',
                        help='if yes, then the multi model will only select half of the second-grade behavior')      
    parser.add_argument('--buy_loss', nargs='?', default='no',
                        help='if yes, then return the loss of buying behavior')        
    parser.add_argument('--cascade_mode', nargs='?', default='single',
                        help='frozen/pretrain/single')        
    parser.add_argument('--cascade_id', nargs='?', default='0',
                        help='folder id for saving')       
    return parser.parse_args()




def do_eval_job(args, EvalDict):
    global job_num, job_lock, eval_queue, dataset, loss_list, hr_list, ndcg_list

    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    while(1):
        job_num.acquire()
        with job_lock:
            eval_info = eval_queue.get()
        [eval_path, epoch_count, batch_time, train_time, train_loss] = eval_info
        hits, ndcgs = [], []
        loader = None
        
        eval_begin = time()
    
        eval_graph = tf.Graph()
        with eval_graph.as_default():
            loader = tf.train.import_meta_graph(eval_path + '.meta', clear_devices = True)

        with tf.Session(graph = eval_graph, config = config) as sess:
            loader.restore(sess, eval_path)

            for idx in xrange(len(EvalDict)):
                if args.model == 'Multi_GMF':
                    gtItem = dataset[0].testRatings[idx][1]
                    predictions = sess.run('loss/inference/score_buy:0', feed_dict = EvalDict[idx])
                else:
                    gtItem = dataset.testRatings[idx][1]
                    predictions = sess.run('loss/inference/output:0', feed_dict = EvalDict[idx])
                rank = 0
                rank_score = predictions[gtItem]
                for i in predictions:
                    if i > rank_score:
                        rank += 1
                if rank < args.topK:
                    hr_tmp = 1
                    ndcg_tmp = math.log(2) / math.log(rank + 2)
                else:
                    hr_tmp = 0
                    ndcg_tmp = 0
                hits.append(hr_tmp)
                ndcgs.append(ndcg_tmp)

        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        eval_time = time() - eval_begin
        
        logging.info(
            "Epoch %d [%.1fs + %.1fs]: train_loss = %.4f  [%.1fs] HR = %.4f, NDCG = %.4f" 
            % (epoch_count + 1, batch_time, train_time, train_loss, eval_time, hr, ndcg))
        print "Epoch %d [%.1fs + %.1fs]: train_loss = %.4f  [%.1fs] HR = %.4f, NDCG = %.4f" % (
            epoch_count + 1, batch_time, train_time, train_loss, eval_time, hr, ndcg)

        # save info to lists 
        hr_list[epoch_count], ndcg_list[epoch_count], loss_list[epoch_count] = (hr, ndcg, train_loss)

        eval_queue.task_done()



def training(model, args, behave_type = None, base_epoch = 0, save = True): 
    global rank_result, loss_list, hr_list, ndcg_list
    
    cascade = args.training_type == 'cascade'
    if cascade:
        cascade_path = None

    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.1

    global pool, dataset, eval_queue, job_num, loss_list, hr_list, ndcg_list

    # initialize for Evaluate
    if args.multiprocess == 'no':
        if args.model in ['Multi_GMF', 'Multi_MLP', 'Multi_NCF']:
            EvalDict = EvalUser.init_evaluate_model(model, dataset[0], args)
        elif args.model == 'FISM':
            pass
        else:
            EvalDict = EvalUser.init_evaluate_model(model, dataset, args)

    with model.g.as_default():
        
        saver = tf.train.Saver()
        if cascade:
            cascade_saver = tf.train.Saver()
            
        with tf.name_scope('optimizer'):
            if args.optimizer == 'Adam':
                if cascade == False:
                    optimizer = tf.train.AdamOptimizer(learning_rate = args.lr).minimize(model.loss)
                else:
                    if behave_type == 'ipv':
                        # optimizer = tf.train.AdamOptimizer(learning_rate = args.lr).minimize(
                        #     model.loss1, var_list = [model.embedding_P, model.embedding_Q, model.h_1])
                        variables = model.variables
                        # g_vars = [var for var in variables]
                    elif behave_type == 'cart':
                        optimizer = tf.train.AdamOptimizer(learning_rate = args.lr).minimize(
                            model.loss2, var_list = [model.W1, model.b1, model.h_2])                    
                    else:
                        optimizer = tf.train.AdamOptimizer(learning_rate = args.lr).minimize(
                            model.loss3, var_list = [model.W2, model.b2, model.h_3])

            elif args.optimizer == 'Adagrad':
                if cascade == False:
                    optimizer = tf.train.AdagradOptimizer(
                        learning_rate = args.lr, initial_accumulator_value = 1e-8).minimize(model.loss)
                else:
                    if behave_type == 'ipv':
                        variables = model.variables
                        g_vars = [var for var in variables]
                        optimizer = tf.train.AdamOptimizer(learning_rate = args.lr).minimize(
                            model.loss1, var_list = g_vars)

                    elif behave_type == 'cart':
                        variables = model.variables

                        if args.cascade_mode == 'frozen':
                            print('cascade_mode: frozen')
                            g_vars = [var for var in variables if 'ipv_para' not in var.name]
                        else:
                            print('cascade_mode: pretrain')
                            g_vars = [var for var in variables]

                        optimizer = tf.train.AdamOptimizer(learning_rate = args.lr).minimize(
                            model.loss2, var_list = g_vars)
                    else:
                        variables = model.variables

                        if args.cascade_mode == 'frozen':
                            print('cascade_mode: frozen')
                            g_vars = [var for var in variables if ('ipv_para' not in var.name) and ('cart_para' not in var.name)]
                        else:
                            print('cascade_mode: pretrain')
                            g_vars = [var for var in variables]

                        optimizer = tf.train.AdamOptimizer(learning_rate = args.lr).minimize(
                            model.loss3, var_list = g_vars)

    with tf.Session(graph = model.g, config = config) as sess:
        # initial training
        # if ! (cascade and (behave_type == 'cart' or behave_type == 'buy')):
        sess.run(tf.global_variables_initializer())
        if cascade and (behave_type == 'cart' or behave_type == 'buy'):
            cascade_path = args.model + args.cascade_id + '/' + args.model + '_'
            # cascade_path = '/data3/gaochen/gandahua/MBR/model_save/cascade/' + filename + '/' + args.model + '_'     
            if behave_type == 'cart':
                cascade_path += 'ipv'
            elif behave_type == 'buy':
                cascade_path += 'cart'
            print('Load data from' + cascade_path)
            cascade_saver.restore(sess, cascade_path)

        logging.info("--- Start training ---")
        print("--- Start training ---")

        # plot network
        print('plot_network:', args.plot_network)
        if args.plot_network:
            print "writing network to TensorBoard/graphs/network"
            writer = tf.summary.FileWriter('./TensorBoard/graphs/network')
            writer.add_graph(sess.graph)
            return 0

        # dict for printing behavior type
        b_dict = {}
        b_dict['vb'], b_dict['cb'], b_dict['vc']= ['view and buy', 'cart and buy', 'view and cart']


        # show initial results
        #hits, ndcgs, ranks = EvalUser.eval(model, sess, dataset, EvalDict, args) 
        #print ('Initial hr: %.4f   ndcg: %.4f'  %(np.mean(hits), np.mean(ndcgs)))

        # train by epoch
        best_hr = 0
        patience = 5
        flag = 0

        for epoch_count in range(args.epochs):
            
            batch_begin = time()
            if 'Multi' in args.model or 'BPR' in args.model or 'CMF' in args.model:
                if args.buy_loss == 'no':
                    samples = BatchUser.sampling_3(args, dataset, args.num_neg)
                else:
                    samples = BatchUser.sampling_3(args, dataset, args.num_neg)[:4]
                    if args.buy_loss == 'yes':
                        buy_samples = BatchUser.sampling_3(args, dataset, args.num_neg)[4:]

                if args.b_num == 3:
                    bs = args.batch_size
                else:
                    if args.b_2_type == 'cb':
                        bs = args.batch_size_cart
                    else:
                        bs = args.batch_size_ipv

                # samples = pkl.load(open('/data3/gaochen/gandahua/MBR/batch_sample/multi_data_ep_%d.pkl' %(epoch_count), 'rb'))
                batches = BatchUser.shuffle_3(samples, bs, args)

                if args.buy_loss == 'yes':
                    buy_batches = BatchUser.unshuffle(buy_samples, bs)
            
                print('Already generate batch, behavior is %d(%s), \n\
                       batch size is %d, all training entries: %d' % (args.b_num, b_dict[args.b_2_type], bs, len(samples[0])))
                batch_time = time() - batch_begin


                train_begin = time()

                if args.buy_loss == 'yes':
                    train_loss = training_batch_3(model, sess, batches, args, optimizer, buy_batches=buy_batches)
                else:
                    if cascade == True:
                        train_loss = training_batch_3(model, sess, batches, args, optimizer, behave_type=behave_type)
                    else:
                        train_loss = training_batch_3(model, sess, batches, args, optimizer)

                train_time = time() - train_begin


            else:
                samples = BatchUser.sampling(args, dataset, args.num_neg)
                print('all training number: %d' % len(samples[0]))
                # print('first label: %s' % samples[2][:20])
                # print('first user: %s' % samples[0][:20])
                # print('first item: %s' % samples[1][:20])                                
                if args.dataset == 'bb2':
                    # samples = pkl.load(open('/data3/gaochen/gandahua/MBR/batch_sample/single_data_ep_%d_cs.pkl' %(epoch_count), 'rb'))
                    if behave_type == 'ipv':
                        bs = int(args.batch_size_ipv/4)
                        batches = BatchUser.shuffle(samples, bs, args)
                    elif behave_type == 'cart':
                        bs = int(args.batch_size_cart/4)
                        batches = BatchUser.shuffle(samples, bs, args)
                    else:
                        bs = int(args.batch_size_buy/4)
                        batches = BatchUser.shuffle(samples, bs, args)

                else:
                    # samples = pkl.load(open('/data3/gaochen/gandahua/MBR/batch_sample/single_data_ep_%d.pkl' %(epoch_count), 'rb'))
                    if behave_type == 'ipv':
                        bs = args.batch_size_ipv
                        batches = BatchUser.shuffle(samples, bs, args)
                    elif behave_type == 'cart':
                        bs = args.batch_size_cart
                        batches = BatchUser.shuffle(samples, bs, args)
                    else:
                        bs = args.batch_size_buy
                        batches = BatchUser.shuffle(samples, bs, args)
                
                print('Already generate batch, batch size is %d' % bs)       
                batch_time = time() - batch_begin
                train_begin = time()
                if cascade == True:
                    train_loss = training_batch(model, sess, batches, args, optimizer, behave_type)
                else:
                    train_loss = training_batch(model, sess, batches, args, optimizer)
                train_time = time() - train_begin
                print('train time: %d' % train_time)

            if epoch_count % args.verbose == 0 and args.evaluate == 'yes':
                if args.multiprocess == 'yes':
                    # save model
                    eval_path = str(epoch_count) + '_' + filename
                    saver.save(sess, eval_path)

                    if epoch_count == (args.epochs - 1):
                        model_save_path = filename
                        saver.save(sess, model_save_path)
                    eval_info = [eval_path, epoch_count, batch_time, train_time, train_loss]
                    eval_queue.put(eval_info)
                    job_num.release()

                else:
                    eval_begin = time()
                    if cascade == True:
                        hits50,hits100,hits200, ndcgs50,ndcgs100,ndcgs200,ranks = EvalUser.eval(model, sess, dataset, EvalDict, args, behave_type)
                    elif args.model == 'FISM':
                        hits, ndcgs, ranks = EvalUser.eval_FISM(model, sess, dataset, args)
                    else:
                        hits50,hits100,hits200, ndcgs50,ndcgs100,ndcgs200,ranks  = EvalUser.eval(model, sess, dataset, EvalDict, args)
                    hr50, ndcg50 = np.array(hits50).mean(), np.array(ndcgs50).mean()
                    hr100, ndcg100 = np.array(hits100).mean(), np.array(ndcgs100).mean()
                    hr200, ndcg200 = np.array(hits200).mean(), np.array(ndcgs200).mean()
                    eval_time = time() - eval_begin
                    logging.info(
                        "Epoch %d [%.1fs + %.1fs]: train_loss = %.4f  [%.1fs] HR50 = %.4f, NDCG50 = %.4f, HR100 = %.4f, NDCG100 = %.4f HR200 = %.4f, NDCG200 = %.4f" 
                        % (epoch_count + 1, batch_time, train_time, train_loss, eval_time, hr50, ndcg50,hr100, ndcg100,hr200, ndcg200))
                    print "Epoch %d [%.1fs + %.1fs]: train_loss = %.4f  [%.1fs] HR50 = %.4f, NDCG50 = %.4f, HR100 = %.4f, NDCG100 = %.4f HR200 = %.4f, NDCG200 = %.4f" % (epoch_count + 1, batch_time, train_time, train_loss, eval_time, hr50, ndcg50,hr100, ndcg100,hr200, ndcg200)
                    
                    # save results, save model
                    (hr_list[base_epoch + epoch_count], ndcg_list[base_epoch + epoch_count], 
                        loss_list[base_epoch + epoch_count]) = (hr100, ndcg100, train_loss)
                    rank_result.append(ranks)
                    

                    if cascade:
                        best_hr = max(best_hr, hr100)
                        if best_hr == hr100:
                            flag = 0
                        else:
                            flag += 1
                        if flag > patience:
                            print('\n\n\n\n Overall %d epochs for %s' %(epoch_count+1, behave_type))
                            logging.info('--- Overall %d epochs for %s ---' %(epoch_count+1, behave_type))
                            break

        if cascade and (behave_type == 'ipv' or behave_type == 'cart'):
            cascade_path =  args.model + args.cascade_id + '/'
            # cascade_path = '/data3/gaochen/gandahua/MBR/model_save/cascade/' + filename + '/'    
            if not os.path.exists(cascade_path):
                os.makedirs(cascade_path)
            cascade_saver.save(sess, cascade_path + args.model + '_' + behave_type)
            print('save data into' + cascade_path + args.model + '_' + behave_type)



# recover results from saved models
def eval_from_saved_model(model, args):
    global hr_recover, ndcg_recover
    
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if args.model == 'Multi_GMF':
        EvalDict = EvalUser.init_evaluate_model(model, dataset[0])
    else:
        EvalDict = EvalUser.init_evaluate_model(model, dataset)
    
    hits, ndcgs = [], []
    loader = None

    eval_begin = time()
    eval_graph = tf.Graph()
    with eval_graph.as_default():
        model_load_path =  filename
        loader = tf.train.import_meta_graph(model_load_path + '.meta', clear_devices = True)

    with tf.Session(graph = eval_graph, config = config) as sess:
        loader.restore(sess, model_load_path)

        for idx in xrange(len(EvalDict)):
            if args.model == 'Multi_GMF':
                gtItem = dataset[0].testRatings[idx][1]
                predictions = sess.run('loss/inference/score_buy:0', feed_dict = EvalDict[idx])
            else:
                gtItem = dataset.testRatings[idx][1]
                predictions = sess.run('loss/inference/output:0', feed_dict = EvalDict[idx])
            rank = 0
            rank_score = predictions[gtItem]
            print len(dataset[0].trainRatings[idx])
            for i in predictions:
                if i > rank_score:
                    rank += 1
            if rank < args.topK:
                hr_tmp = 1
                ndcg_tmp = math.log(2) / math.log(rank + 2)
            else:
                hr_tmp = 0
                ndcg_tmp = 0
            hits.append(hr_tmp)
            ndcgs.append(ndcg_tmp)

        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()

        print "Final HR = %.4f, NDCG = %.4f" % (hr, ndcg)
        
        # save info to the server 
        hr_recover, ndcg_recover = hr, ndcg



# used for single task learning
def training_batch(model, sess, batches, args, optimizer, behave_type=None):

    loss_train = 0.0
    loss_train_all = 0.0

    if args.model == 'FISM':
        user_input, item_input, item_rate, item_num, labels = batches
        num_batch = len(batches[1])
        # print('shape of item rate: (%d, %d)' %(item_rate[0].shape[0], item_rate[0].shape[1]))
        # print('first item rate: %s' % item_rate[0])
        for i in range(len(user_input)):
            feed_dict = {model.user_input: user_input[i][:, None],
                         model.item_input: item_input[i][:, None],
                         model.item_rate: item_rate[i],
                         model.item_num: np.reshape(item_num[i], [-1, 1]),
                         model.labels: np.reshape(labels[i], [-1, 1])}
            _, loss_train = sess.run([optimizer, model.loss], feed_dict)
            loss_train_all += loss_train
    else:
        loss_no_reg_all = 0.0
        loss_reg_all = 0.0

        user_input, item_input, labels = batches
        num_batch = len(batches[1])
        for i in range(len(labels)):
            feed_dict = {model.user_input: user_input[i][:, None],
                         model.item_input: item_input[i][:, None],
                         model.labels: labels[i][:, None]}
            if behave_type == 'ipv':
                _, loss_train = sess.run([optimizer, model.loss1], feed_dict)
            elif behave_type == 'cart':
                _, loss_train = sess.run([optimizer, model.loss2], feed_dict)
            elif behave_type == 'buy':
                _, loss_train = sess.run([optimizer, model.loss3], feed_dict)
            else:
                if args.buy_loss == 'no':
                    _, loss_train = sess.run([optimizer, model.loss], feed_dict)
                else:
                    _, loss_train = sess.run([optimizer, model.loss_buy], feed_dict)

            if args.model == 'pure_GMF':
                loss_no_reg, loss_reg = sess.run([model.loss_no_reg, model.loss_reg], feed_dict)
                loss_reg_all += loss_reg
                loss_no_reg_all += loss_no_reg


            loss_train_all += loss_train

        if args.model == 'pure_GMF':
            print('loss_no_reg:%.6f, loss_reg:%.6f' %(loss_no_reg_all/num_batch, loss_reg_all/num_batch))
    return loss_train_all / num_batch         



# used for multitask-learning
def training_batch_3(model, sess, batches, args, optimizer, buy_batches=None, behave_type=None):
    
    loss_train = 0.0
    loss_train_all = 0.0

    if 'BPR' in args.model:
        user_input, item_input, item_input_neg = batches
        num_batch = len(batches[1])
        for i in range(len(user_input)):
            feed_dict = {model.user_input: np.reshape(user_input[i], [-1, 1]),
                         model.item_input: np.reshape(item_input[i], [-1, 1]),
                         model.item_input_neg: np.reshape(item_input_neg[i], [-1, 1])}
            _, loss_train = sess.run([optimizer, model.loss], feed_dict)
            loss_train_all += loss_train

    elif 'CMF' in args.model:
        (user_input, item_input_buy, item_input_cart, item_input_ipv,
            labels_buy, labels_cart, labels_ipv) = batches
        num_batch = len(user_input)
        for i in range(len(user_input)):
            feed_dict = {model.user_input: np.reshape(user_input[i], [-1, 1]),
                         model.item_input_buy: np.reshape(item_input_buy[i], [-1, 1]),
                         model.item_input_cart: np.reshape(item_input_cart[i], [-1, 1]),
                         model.item_input_ipv: np.reshape(item_input_ipv[i], [-1, 1]),
                         model.labels_buy: np.reshape(labels_buy[i], [-1, 1]),
                         model.labels_cart: np.reshape(labels_cart[i], [-1, 1]),
                         model.labels_ipv: np.reshape(labels_ipv[i], [-1, 1])}

            _, loss_train = sess.run([optimizer, model.loss], feed_dict)
            loss_train_all += loss_train

    else:
        user_input, item_input, labels = batches
        num_batch = len(batches[1])
        for i in range(len(labels)):
            if args.b_num == 2:
                if args.b_2_type == 'vc':           
                    feed_dict = {model.user_input: user_input[i][:, None],
                                 model.item_input: item_input[i][:, None],
                                 model.labels_ipv: labels[i][:, 2][:, None],
                                 model.labels_cart: labels[i][:, 1][:, None]}
                    _, loss_train = sess.run([optimizer, model.loss], feed_dict)  
                
                elif args.b_2_type == 'vb':
                    feed_dict = {model.user_input: user_input[i][:, None],
                                 model.item_input: item_input[i][:, None],
                                 model.labels_ipv: labels[i][:, 2][:, None],
                                 model.labels_buy: labels[i][:, 1][:, None]}
         
                    if behave_type == 'ipv':
                        _, loss_train = sess.run([optimizer, model.loss1], feed_dict)  
                    elif behave_type == 'cart':
                        _, loss_train = sess.run([optimizer, model.loss2], feed_dict)  
                    else:
                        _, loss_train = sess.run([optimizer, model.loss], feed_dict)  

                
                else:
                    feed_dict = {model.user_input: user_input[i][:, None],
                                 model.item_input: item_input[i][:, None],
                                 model.labels_cart: labels[i][:, 2][:, None],
                                 model.labels_buy: labels[i][:, 1][:, None]}   
                    _, loss_train = sess.run([optimizer, model.loss], feed_dict)  

                                       
           
            else:
                feed_dict = {model.user_input: user_input[i][:, None],
                             model.item_input: item_input[i][:, None],
                             model.labels_ipv: labels[i][:, 2][:, None],
                             model.labels_cart: labels[i][:, 1][:, None],
                             model.labels_buy: labels[i][:, 0][:, None]}

                if behave_type == 'ipv':
                    _, loss_train = sess.run([optimizer, model.loss1], feed_dict)
                elif behave_type == 'cart':
                    _, loss_train = sess.run([optimizer, model.loss2], feed_dict)
                elif behave_type == 'buy':
                    _, loss_train = sess.run([optimizer, model.loss3], feed_dict)
                else:
                    _, loss_train = sess.run([optimizer, model.loss], feed_dict) 

            
            loss_train_all += loss_train

        if args.buy_loss == 'yes':
            print('[loss mode]   Buy_loss')
            print('batch_num: %d' %(len(buy_batches[0])))
            print('batch_size: %d' %(len(buy_batches[0][0])))
            print('batch_user: %s' %(buy_batches[0][0][-20:]))
            print('batch_item: %s' %(buy_batches[1][0][-20:]))
            print('batch_ipv: %s' %(buy_batches[2][0][-20:]))
            print('batch_cart: %s' %(buy_batches[3][0][-20:]))
            print('batch_buy: %s' %(buy_batches[4][0][-20:]))

            print('batch_user2: %s' %(buy_batches[0][1][:4]))
            print('batch_item2: %s' %(buy_batches[1][1][:4]))
            print('batch_ipv2: %s' %(buy_batches[2][1][:4]))
            print('batch_cart2: %s' %(buy_batches[3][1][:4]))
            print('batch_buy2: %s' %(buy_batches[4][1][:4]))

            loss_train_all = 0.0
            for i in range(len(buy_batches[0])):
                feed_dict_buy = {model.user_input: np.array(buy_batches[0][i]).reshape(-1, 1),
                                 model.item_input: np.array(buy_batches[1][i]).reshape(-1, 1),
                                 model.labels_ipv: np.array(buy_batches[2][i]).reshape(-1, 1),
                                 model.labels_cart: np.array(buy_batches[3][i]).reshape(-1, 1),
                                 model.labels_buy: np.array(buy_batches[4][i]).reshape(-1, 1)}

                loss_train = sess.run(model.loss_buy, feed_dict_buy)
                loss_train_all += loss_train
    
    if args.buy_loss == 'yes':
        return loss_train_all / len(buy_batches[0])

    else:      
        return loss_train_all / num_batch   


# def validate_batch(model, sess, batches, args):
# input: model, sess, batches
# output: training_loss
# def training_loss(model, sess, batches, args):
#     train_loss = 0.0
#     num_batch = len(batches[1])
#     user_input, item_input, labels = batches
#     if args.loss_func == "logloss":
#         for i in range(len(labels)):
#             feed_dict = {model.user_input: user_input[i][:, None],
#                          model.item_input: item_input[i][:, None],
#                          model.labels: labels[i][:, None]}
#             train_loss += sess.run(model.loss, feed_dict)
#     else:
#         for i in range(len(labels)):
#             feed_dict = {model.user_input: user_input[i][:, None],
#                          model.item_input: item_input[i],
#                          model.labels: labels[i][:, None]}
#             loss = sess.run(model.loss, feed_dict)
#             # train_loss += sess.run(model.loss, feed_dict)
#             train_loss += loss
#     return train_loss / num_batch


def init_logging_and_result(args):
    global filename
    path_log = 'Log' 
    if not os.path.exists(path_log):
        os.makedirs(path_log)

    # define factors
    F_model = args.model
    F_dataset = args.dataset
    F_embedding = args.embed_size
    F_topK = args.topK
    F_layer_num = args.layer_num
    F_num_neg = args.num_neg
    F_trail_id = args.trial_id
    F_optimizer = args.optimizer + str(args.lr)
    F_loss_weight = args.loss_coefficient
    F_beta = args.beta
    F_alpha = args.alpha
    F_en_MC = args.en_MC
    F_dropout = args.dropout
    F_reg = args.regs
    F_b_num = args.b_num
    F_b_2_type = args.b_2_type
    F_half = args.half_behave
    F_buy_loss = args.buy_loss

    if args.training_type == 'cascade':
        F_cascade = 'C'
        F_cascade_mode = args.cascade_mode

    else:
        F_cascade = 'X'
        F_cascade_mode = 'X'


    if F_model not in ['pure_NCF', 'pure_MLP', 'Multi_NCF', 'Multi_MLP', 'GMF_FC', 'NCF_FC']:
        F_layer_num = 'X'
    if F_model not in ['Multi_MLP', 'Multi_NCF', 'Multi_GMF']:
        F_b_2_type = 'X'
    if (F_model != 'Multi_BPR'):
        F_dropout = 'X'
    if (F_model != 'Multi_BPR') and (F_en_MC != 'yes'):
        F_beta = 'X'
    if F_num_neg == 4:
        F_num_neg = 'D'
    # if F_optimizer == 'Adagrad0.01':
    #     F_optimizer = 'D'
    if F_loss_weight == '[1/3,1/3,1/3]':
        F_loss_weight = 'D'
    else:
        F_loss_weight = F_loss_weight.replace('/', '-')
    if F_model != 'FISM':
        F_alpha = 'X'
    if F_b_num == 3:
        F_b_2_type = 'X'

    if F_half == 'no':
        if F_buy_loss == 'no':
            filename = "log-%s-%s-%s-%s-%s-%s-%s-%s-%s-%s-%s-%s-b-%s-a-%s-%s%s" %(
                       F_model, F_dataset, F_embedding, F_topK, F_layer_num, F_num_neg, F_loss_weight,\
                       F_optimizer, F_trail_id, F_beta, F_dropout, F_reg, F_b_2_type, F_alpha, F_cascade, F_cascade_mode)
        else:
            filename = "log-%s-%s-%s-%s-%s-%s-%s-%s-%s-%s-%s-%s-b-%s-a-%s-bloss-%s%s" %(
                       F_model, F_dataset, F_embedding, F_topK, F_layer_num, F_num_neg, F_loss_weight,\
                       F_optimizer, F_trail_id, F_beta, F_dropout, F_reg, F_b_2_type, F_alpha, F_cascade, F_cascade_mode)            
    else:
        filename = "log-%s-%s-%s-%s-%s-%s-%s-%s-%s-%s-%s-%s-b-%s-a-%s-half-%s%s" %(
                   F_model, F_dataset, F_embedding, F_topK, F_layer_num, F_num_neg, F_loss_weight,\
                   F_optimizer, F_trail_id, F_beta, F_dropout, F_reg, F_b_2_type, F_alpha, F_cascade, F_cascade_mode)        

    
    logging.basicConfig(filename=path_log+'/'+filename, level=logging.INFO) 
    logging.info('Use Multiprocess to Evaluate: %s' %args.multiprocess)


def save_results(args, cascade = False):
    if args.recover == 'yes':
        path_result = 'Recover'
    else:
        path_result = 'Result'

    if not os.path.exists(path_result):
        os.makedirs(path_result)

    if args.recover == 'yes':
        with open(path_result+'/'+filename, 'w') as output:
            output.write('HR:%.4f,NDCG:%.4f' %(hr_recover, ndcg_recover))       
    else:
        if cascade:
            pass

        else:
            with open(path_result+'/'+filename, 'w') as output:
                for i in range(len(loss_list)):
                    output.write('%.4f,%.4f,%.4f\n' %(loss_list[i], hr_list[i], ndcg_list[i]))
            rank_path = ''
          

if __name__ == '__main__':
    args = parse_args()

    dataset = None
    filename = None
    hr_recover = None
    ndcg_recover = None
    eval_queue = JoinableQueue()
    job_num = Semaphore(0)
    job_lock = Lock()

    rank_result = []
    if 'FC' in args.model:
        loss_list = range(3 * args.epochs)
        hr_list = range(3 * args.epochs)
        ndcg_list = range(3 * args.epochs)
    else:
        loss_list = range(args.epochs)
        hr_list = range(args.epochs)
        ndcg_list = range(args.epochs)

    # initialize logging and configuration
    print('------ %s ------' %(args.process_name))
    #setproctitle.setproctitle(args.process_name)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    init_logging_and_result(args)


    # load data 
    print('--- data generation start ---')
    data_gen_begin = time()

    if args.dataset == 'bb1':
        print('load bb1 data')
        path = '../data/Beibei/beibei'
    elif args.dataset == 'bb2':
        print('load bb2 data')
        path = '/data3/gaochen/gandahua/Data/'
    elif args.dataset == 'bb3':
        print('load bb3 data')
        pass
    elif args.dataset == 'ali':
        print('load ali data')
        path = '../data/taobao/taobao'
    elif args.dataset == 'ali2':
        print('load ali 2 data')
        path = '/home/stu/gandahua/MBR/Data/ali'
    elif args.dataset == 'ali3':
        print('load ali 3 data')
        path = '/home/stu/gandahua/MBR/Data_ali/ali'    
    else:
        pass

    if ('BPR' in args.model) or (args.en_MC == 'yes') or (args.model == 'FISM') or ('CMF' in args.model):
        dataset_all = Dataset(path = path, load_type = 'dict')
    else:
        dataset_ipv = Dataset(path = path, b_type = 'ipv', en_half=args.half_behave)
        dataset_cart = Dataset(path = path, b_type = 'cart', en_half=args.half_behave)
        dataset_buy = Dataset(path = path, b_type = 'buy', en_half=args.half_behave)
        dataset_all = (dataset_ipv, dataset_cart, dataset_buy)

    print('data generation [%.1f s]' %(time()-data_gen_begin))

    # model training and evaluating 
    if args.model == 'Multi_GMF':
        model = Multi_GMF(dataset_all[0].num_users, dataset_all[0].num_items, args)
        print('num_users:%d   num_items:%d' %(dataset_ipv.num_users, dataset_ipv.num_items))
        model.build_graph()
        dataset = dataset_all

        # recover result or not
        if args.recover == 'yes':
            eval_from_saved_model(model, args)

        else:
            if args.multiprocess == 'yes':
                print('start multiprocess')
                train_process = Process(target = training, args = (model, args))
                train_process.start()
                # evaluate
                # initialize for Evaluate
                EvalDict = EvalUser.gen_feed_dict(dataset[0])
            
                cpu_num = 3
                eval_pool = Pool(cpu_num)
                for _ in range(cpu_num):
                    eval_pool.apply_async(do_eval_job, (args, EvalDict))
                train_process.join()
                eval_queue.close()
                eval_queue.join()

            else:
                print('start single process')
                if args.training_type == 'independent':
                    training(model, args)
                
                else:
                    training(model, args, behave_type='ipv')
                    rank_result = []
                    loss_list = range(args.epochs)
                    hr_list = range(args.epochs)
                    ndcg_list = range(args.epochs)

                    model.build_graph()
                    training(model, args, behave_type='cart')
                    rank_result = []
                    loss_list = range(args.epochs)
                    hr_list = range(args.epochs)
                    ndcg_list = range(args.epochs)

                    model.build_graph()
                    training(model, args, behave_type='buy')
                    rank_result = []
                    loss_list = range(args.epochs)
                    hr_list = range(args.epochs)
                    ndcg_list = range(args.epochs)



    elif args.model == 'Multi_MLP':
        model = Multi_MLP(dataset_all[0].num_users, dataset_all[0].num_items, args)
        print('num_users:%d   num_items:%d' %(dataset_ipv.num_users, dataset_ipv.num_items))
        model.build_graph()
        dataset = dataset_all

        # recover result or not
        if args.recover == 'yes':
            eval_from_saved_model(model, args)

        else:
            if args.multiprocess == 'yes':
                print('start multiprocess')
                train_process = Process(target = training, args = (model, args))
                train_process.start()
                # evaluate
                # initialize for Evaluate
                EvalDict = EvalUser.gen_feed_dict(dataset[0])
            
                cpu_num = 3
                eval_pool = Pool(cpu_num)
                for _ in range(cpu_num):
                    eval_pool.apply_async(do_eval_job, (args, EvalDict))
                train_process.join()
                eval_queue.close()
                eval_queue.join()

            else:
                print('start single process')
                if args.training_type == 'independent':
                    training(model, args)
                
                else:
                    training(model, args, behave_type='ipv')
                    rank_result = []
                    loss_list = range(args.epochs)
                    hr_list = range(args.epochs)
                    ndcg_list = range(args.epochs)

                    model.build_graph()
                    training(model, args, behave_type='cart')
                    rank_result = []
                    loss_list = range(args.epochs)
                    hr_list = range(args.epochs)
                    ndcg_list = range(args.epochs)

                    model.build_graph()
                    training(model, args, behave_type='buy')
                    rank_result = []
                    loss_list = range(args.epochs)
                    hr_list = range(args.epochs)
                    ndcg_list = range(args.epochs)



    elif args.model == 'Multi_NCF':
        # model = Multi_NCF(dataset_all[0].num_users, dataset_all[0].num_items, args)
        model = Multi_NCF_2(dataset_all[0].num_users, dataset_all[0].num_items, args)        
        print('num_users:%d   num_items:%d' %(dataset_ipv.num_users, dataset_ipv.num_items))
        model.build_graph()
        dataset = dataset_all

        # recover result or not
        if args.recover == 'yes':
            eval_from_saved_model(model, args)

        else:
            if args.multiprocess == 'yes':
                print('start multiprocess')
                train_process = Process(target = training, args = (model, args))
                train_process.start()
                # evaluate
                # initialize for Evaluate
                EvalDict = EvalUser.gen_feed_dict(dataset[0])
            
                cpu_num = 3
                eval_pool = Pool(cpu_num)
                for _ in range(cpu_num):
                    eval_pool.apply_async(do_eval_job, (args, EvalDict))
                train_process.join()
                eval_queue.close()
                eval_queue.join()

            else:
                print('start single process')
                if args.training_type == 'independent':
                    training(model, args)
                
                else:
                    training(model, args, behave_type='ipv')
                    rank_result = []
                    loss_list = range(args.epochs)
                    hr_list = range(args.epochs)
                    ndcg_list = range(args.epochs)

                    model.build_graph()
                    training(model, args, behave_type='cart')
                    rank_result = []
                    loss_list = range(args.epochs)
                    hr_list = range(args.epochs)
                    ndcg_list = range(args.epochs)

                    if args.b_num == 3:
                        model.build_graph()
                        training(model, args, behave_type='buy')
                        rank_result = []
                        loss_list = range(args.epochs)
                        hr_list = range(args.epochs)
                        ndcg_list = range(args.epochs)



    elif args.model == 'pure_GMF':
        if args.en_MC == 'yes':
            dataset = dataset_all
        else:
            dataset = dataset_buy
        model = pure_GMF(dataset.num_users, dataset.num_items, args)        
        print('num_users:%d   num_items:%d' %(dataset.num_users, dataset.num_items))
        model.build_graph()

        # recover result or not
        if args.recover == 'yes':
            eval_from_saved_model(model, args)

        else:
            if args.multiprocess == 'yes':
                pass
            else:
                print('start single process')
                training(model, args, behave_type='buy')
                # training(model, args, behave_type='buy')


    elif args.model == 'pure_MLP':
        if args.en_MC == 'yes':
            dataset = dataset_all
        else:
            dataset = dataset_buy

        model = pure_MLP(dataset.num_users, dataset.num_items, args)        
        print('num_users:%d   num_items:%d' %(dataset.num_users, dataset.num_items))
        model.build_graph()

        # recover result or not
        if args.recover == 'yes':
            eval_from_saved_model(model, args)

        else:
            if args.multiprocess == 'yes':
                pass
            else:
                print('start single process')
                training(model, args, behave_type='buy')


    elif args.model == 'pure_NCF':
        if args.en_MC == 'yes':
            dataset = dataset_all
        else:
            dataset = dataset_buy

        # model = pure_NCF(dataset.num_users, dataset.num_items, args)        
        model = pure_NCF_2(dataset.num_users, dataset.num_items, args)        
        print('num_users:%d   num_items:%d' %(dataset.num_users, dataset.num_items))
        model.build_graph()

        # recover result or not
        if args.recover == 'yes':
            eval_from_saved_model(model, args)

        else:
            if args.multiprocess == 'yes':
                pass
            else:
                print('start single process')
                training(model, args, behave_type='buy')


    elif args.model == 'FISM':
        model = FISM(dataset_all.num_items, dataset_all.num_users, dataset_all.max_rate, args)
        print('num_users:%d   num_items:%d  max_rate:%d' %(
               dataset_all.num_users, dataset_all.num_items, dataset_all.max_rate))
        model.build_graph()
        dataset = dataset_all

        # recover result or not
        if args.recover == 'yes':
            eval_from_saved_model(model, args)
        
        else:
            if args.multiprocess == 'yes':
                print('start multiprocess')
                train_process = Process(target = training, args = (model, args))
                train_process.start()
                # evaluate
                # initialize for Evaluate
                EvalDict = EvalUser.gen_feed_dict(dataset)
            
                cpu_num = 3
                eval_pool = Pool(cpu_num)
                for _ in range(cpu_num):
                    eval_pool.apply_async(do_eval_job, (args, EvalDict))
                train_process.join()
                eval_queue.close()
                eval_queue.join()

            else:
                print('start single process')
                training(model, args)


    elif args.model == 'CMF':
        model = CMF(dataset_all.num_users, dataset_all.num_items, args)
        print('num_users:%d   num_items:%d' %(dataset_all.num_users, dataset_all.num_items))
        model.build_graph()
        dataset = dataset_all

        print('start single process')
        training(model, args)


    elif 'BPR' in args.model:
        model = BPR(dataset_all.num_users, dataset_all.num_items, args)
        print('num_users:%d   num_items:%d' %(dataset_all.num_users, dataset_all.num_items))
        model.build_graph()
        dataset = dataset_all

        # recover result or not
        if args.recover == 'yes':
            eval_from_saved_model(model, args)
        
        else:
            if args.multiprocess == 'yes':
                print('start multiprocess')
                train_process = Process(target = training, args = (model, args))
                train_process.start()
                # evaluate
                # initialize for Evaluate
                EvalDict = EvalUser.gen_feed_dict(dataset)
            
                cpu_num = 3
                eval_pool = Pool(cpu_num)
                for _ in range(cpu_num):
                    eval_pool.apply_async(do_eval_job, (args, EvalDict))
                train_process.join()
                eval_queue.close()
                eval_queue.join()

            else:
                print('start single process')
                training(model, args)


    elif 'FC' in args.model:
        if args.model == 'GMF_FC':
            model = GMF_FC(dataset_ipv.num_users, dataset_ipv.num_items, args)
        elif args.model == 'NCF_FC':
            model = NCF_FC(dataset_ipv.num_users, dataset_ipv.num_items, args)
        model.build_graph()       
        
        # recover result or not
        if args.recover == 'yes':
            eval_from_saved_model(model, args)
        
        else:
            if args.multiprocess == 'yes':
                print('start multiprocess')
                pass

            else:
                print('start single process')
                dataset = dataset_ipv
                training(model, args, behave_type = 'ipv')
                print('ipv train finished!')
                
                del model
                if args.model == 'GMF_FC':
                    model = GMF_FC(dataset_ipv.num_users, dataset_ipv.num_items, args)
                elif args.model == 'NCF_FC':
                    model = NCF_FC(dataset_ipv.num_users, dataset_ipv.num_items, args) 
                model.build_graph()
                dataset = dataset_cart
                training(model, args, behave_type = 'cart', base_epoch = args.epochs)
                print('cart train finished!')

                del model
                if args.model == 'GMF_FC':
                    model = GMF_FC(dataset_ipv.num_users, dataset_ipv.num_items, args)
                elif args.model == 'NCF_FC':
                    model = NCF_FC(dataset_ipv.num_users, dataset_ipv.num_items, args) 
                model.build_graph()
                dataset = dataset_buy  
                training(model, args, behave_type = 'buy', base_epoch = 2 * args.epochs)
                print('buy train finished!')
    

    save_results(args)





    
