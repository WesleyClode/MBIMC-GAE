# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd
def load_data(csv_file):
    tp = pd.read_csv(csv_file, sep=',')
    return tp

#这里导入你自己的数据
_paths = {
    "taobao":'./log/taobao_2021-05-07 03:54:35/'
    #'Tmall_small':'./raw_data/Tmall_small/tzzs_data.csv'
}
#x_axix，train_pn_dis这些都是长度相同的list()

test = load_data(os.path.join(_paths["taobao"], 'test_loss.csv'))
train = load_data(os.path.join(_paths["taobao"], 'train_loss.csv'))
valid = load_data(os.path.join(_paths["taobao"], 'valid_loss.csv'))


#开始画图
#sub_axix = filter(lambda x:x%200 == 0, x_axix)
train_xx = range(train.shape[0])
test_xx = range(test.shape[0])
valid_xx = range(valid.shape[0])
plt.title('Result Analysis')
print("train[rmse].shape:",train["rmse"].shape)
print("test[rmse].shape:",test["rmse"].shape)
#print("xx:",xx.shape)

plt.plot(train_xx, train["rmse"], color='green', label='training RMSE')
plt.plot(test_xx , test["rmse"], color='red', label='testing RMSE')
plt.plot(valid_xx, valid["rmse"],  color='skyblue', label='valid RMSE')
#plt.plot(x_axix, thresholds, color='blue', label='threshold')
plt.legend() # 显示图例

plt.xlabel('iteration times')
plt.ylabel('RMSE')
plt.show()
plt.savefig('RMSE_negitive3.png')
#python 一个折线图绘制多个曲线