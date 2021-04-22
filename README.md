# MBIMC-GAE

## Requirements

Latest tested combination: Python 3.8.1 + PyTorch 1.4.0 + DGL 0.5.2.

Install [PyTorch](https://pytorch.org/)

Install [DGL](https://github.com/dmlc/dgl)

Other required python libraries: numpy, scipy, pandas, h5py, networkx, tqdm ,bidict etc.

## Data

Supported datasets: Tmall、Tmall_40_4000_40_4000、Tmall_20_4000_20_4000、Tmall_0_4000_20_4000

数据集1. 原数据集：Tmall

```
总体数目: 100150806
item数目: 4162024
user数目: 987994
click:0.002182 %
fav:0.000070 %
cart:0.000134 %
buy:0.000049 %
```

数据集2. Tmall_40_4000_40_4000

选取item的被购买次数大于40，user购买次数大于40的数据。

其中各种行为占比为：

```
总体数目: 211390
item数目: 3947
user数目: 4000
click:1.166912 %
fav:0.034191 %
cart:0.087294 %
buy:0.050532 %
```

数据集3. Tmall_20_4000_20_4000

选取item的被购买次数大于在20的item，从中随机抽取4000种item作为研究对象

选取user中购买次数大于20的user，从中随机抽取4000个人作为样本

```
总体数目: 109460
item数目: 3841
user数目: 4000
click:0.619018 %
fav:0.019266 %
cart:0.048002 %
buy:0.026159 %
```

数据集4：Tmall_0_4000_20_4000


选取item的被购买次数大于在20的item，从中随机抽取4000种item作为研究对象

选取user中购买次数大于0的user，从中随机抽取4000个人作为样本

```
总体数目: 21798
item数目: 3350
user数目: 4000
click:0.139254 %
fav:0.004306 %
cart:0.012246 %
buy:0.006866 %
```

### How to run

```python
python -u train.py --data_name=Tmall_0_4000_20_4000 \
                       --use_one_hot_fea \
                       --gcn_agg_accum=sum \
                       --device 0 \
                       --ARR 0.00000000000 \
                       --train_early_stopping_patience 100 \
                       --layers 2 \
                       --gcn_agg_units 30 \
                       --train_lr 0.01 \
                       --data_valid_ratio 0.1 \
                       --model_activation tanh \
                       --gcn_out_units 30\
                       --sample_rate 3
```

最后的 sample_rate 是说负采样采几个

