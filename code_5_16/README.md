# Inductive Multi-behavior Recommendation with Graph Convolutional Networks

## Requirements

Latest tested combination: Python 3.8.1 + PyTorch 1.4.0 + DGL 0.5.2.

Install [PyTorch](https://pytorch.org/)

Install [DGL](https://github.com/dmlc/dgl)

Other required python libraries: numpy, scipy, pandas, h5py, networkx, tqdm ,bidict etc.

## Data

Supported datasets: taobao, Beibei

### How to run

```python
python -u train.py --data_name=Beibei \
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
                       --sample_rate 5
```

最后的 sample_rate 是说负采样采几个