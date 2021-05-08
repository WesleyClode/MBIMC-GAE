### How to run

To train on taobao, type:
```bash
    python -u train.py --data_name=taobao \
                       --use_one_hot_fea \
                       --gcn_agg_accum=sum \
                       --device 0 \
                       --ARR 0.00000000000 \
                       --train_early_stopping_patience 200 \
                       --layers 2 \
                       --gcn_agg_units 30 \
                       --train_lr 0.01 \
                       --data_valid_ratio 0.1 \
                       --model_activation tanh \
                       --gcn_out_units 30
```

