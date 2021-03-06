'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np
from time import time

class Dataset(object):
    '''
    Loading the data file
        trainMatrix: load rating records as sparse matrix for class Data
        trianList: load rating records as list to speed up user's feature retrieval
        testRatings: load leave-one-out rating test for class Evaluate
        testNegatives: sample the items not rated by user
    '''

    def __init__(self, path, b_type = None, load_type = 'matrix', en_half='no'):
        '''
        Constructor
        '''
        self.b_type = b_type
        self.en_half = en_half
        self.num_users, self.num_items = self.get_users_items_num(path)
        if load_type == 'matrix':
            self.trainMatrix = self.load_training_file_as_matrix(path)

        elif load_type == 'dict':
            self.trainDict = self.load_training_file_as_dict(path)
            max_rate = 0
            for u in self.trainDict.keys():
                rate = len(self.trainDict[u]['buy'])
                max_rate = max(rate, max_rate)
            self.max_rate = max_rate

        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        print "already load the testRatings..." 
        return ratingList

    def load_training_file_as_dict(self, filename):
        
        filename_ipv = filename + '-ipv.train.rating'
        filename_cart = filename + '-cart.train.rating'
        filename_buy = filename + '-buy.train.rating'

        trainDict = {}
        for i in range(self.num_users):
            trainDict[i] = {}
            trainDict[i]['ipv'], trainDict[i]['cart'], trainDict[i]['buy'] = [], [], []

        with open(filename_ipv, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                
                trainDict[u]['ipv'].append(i)
                line = f.readline()

        with open(filename_cart, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i= int(arr[0]), int(arr[1])
                
                trainDict[u]['cart'].append(i)
                line = f.readline()

        with open(filename_buy, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                
                trainDict[u]['buy'].append(i)
                line = f.readline()

        for i in trainDict.keys():
            if len(trainDict[i]['ipv']) == 0 and \
                len(trainDict[i]['cart']) == 0 and \
                len(trainDict[i]['buy']) == 0:
                del trainDict[i]

        print "already load the trainDict..." 
        return trainDict

    def get_users_items_num(self, filename):

        filename_ipv = filename + '-ipv.train.rating'
        filename_cart = filename + '-cart.train.rating'
        filename_buy = filename + '-buy.train.rating'

        num_users_ipv, num_items_ipv = 0, 0
        num_users_cart, num_items_cart = 0, 0
        num_users_buy, num_items_buy = 0, 0

        with open(filename_ipv, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users_ipv = max(num_users_ipv, u)
                num_items_ipv = max(num_items_ipv, i)
                line = f.readline()

        with open(filename_cart, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users_cart = max(num_users_cart, u)
                num_items_cart = max(num_items_cart, i)
                line = f.readline()

        with open(filename_buy, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users_buy = max(num_users_buy, u)
                num_items_buy = max(num_items_buy, i)
                line = f.readline()                       

        # Construct matrix
        return max(num_users_ipv, num_users_cart, num_users_buy) + 1, \
            max(num_items_ipv, num_items_cart, num_items_buy) + 1

    def load_training_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        if self.en_half == 'no':
            if self.b_type == 'ipv':
                filename = filename + '-ipv.train.rating'
            elif self.b_type == 'cart':
                filename = filename + '-cart.train.rating'
            elif self.b_type == 'buy':
                filename = filename + '-buy.train.rating'
            
            mat = sp.dok_matrix((self.num_users, self.num_items), dtype = np.float32)

            with open(filename, "r") as f:
                line = f.readline()
                while line != None and line != "":
                    arr = line.split("\t")
                    user, item = int(arr[0]), int(arr[1])
                    mat[user, item] = 1.0
                    line = f.readline()
            print "already load the trainMatrix..." 

        else:
            # create matrix 
            mat = sp.dok_matrix((self.num_users, self.num_items), dtype = np.float32)
            # create dict
            trainDict = self.load_training_file_as_dict(filename)
            # write new matrix
            for i in range(len(trainDict)):
                user = i
                if self.b_type == 'ipv':
                    bar = len(trainDict[i]['ipv']) // 2
                    item_ipv = np.setdiff1d(trainDict[i]['ipv'], trainDict[i]['buy'])
                    if len(item_ipv) > bar:
                        item_keep = np.random.choice(item_ipv, bar, replace=False).tolist() + trainDict[i]['buy']
                    else:
                        item_keep = trainDict[i]['buy']
                elif self.b_type == 'cart':
                    bar = len(trainDict[i]['cart']) // 2
                    item_cart = np.setdiff1d(trainDict[i]['cart'], trainDict[i]['buy'])
                    if len(item_cart) > bar:
                        item_keep = np.random.choice(item_cart, bar, replace=False).tolist() + trainDict[i]['buy']
                    else:
                        item_keep = trainDict[i]['buy']
                else:
                    item_keep = trainDict[i]['buy']
                
                for item in item_keep:
                    mat[user, item] = 1.0

            print "already load the trainMatrix(half)..." 
        
        return mat


