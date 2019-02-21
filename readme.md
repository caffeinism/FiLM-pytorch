# Pytorch implementation of [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871)

## Requirements

- Python3
- Pytorch 1.0.0
- TensorBoardX

## usage

train
```
python train.py 
    --batch_size [64]
    --n_epoch [120]
    --lr [1e-4]
    --weight_decay [1e-4]
    --save_dir [model]
    --dataset [data/sort-of-clevr.pickle]
    --init [kaiming]
    --resume []
    --n_res [6]
    --seed [12345]
    --n_cpu [4]
```

test
```
python test.py
    --n_res
    --dataset
    --model
```