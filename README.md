# Spatial Transformer Networks

Chainer implementations of *Spatial Transformer Networks*  [https://arxiv.org/abs/1506.02025](https://arxiv.org/abs/1506.02025).


This implementation tries to reproduce the distorted MNIST dataset described in the original paper.

## Transformation Grid Sample

An animation of the transformation grids from iteration 0 to 200 using a batch size of 128.

![](example/grids.gif)

Loss and accuracy plots of the ST-CNN model, compared to a CNN without the spatial transformer (ST) layer, *CNN(Pooling)*. Since the first layer becomes an average pooling layer, we also plot the training curves of a CNN without both the ST layer and the average pooling operation, *CNN*.

![](example/plot.png)

## Train

```bash
python train.py --max-iter 1000 --out result --gpu 0
```
