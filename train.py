import matplotlib  # NOQA
matplotlib.use('Agg')  # NOQA

import argparse

from chainer import datasets
from chainer import iterators
from chainer import links as L
from chainer import optimizers
from chainer import training
from chainer.training import extensions
from chainer.training import Trainer
from chainercv.datasets import TransformDataset

import numpy as np
from skimage import transform

from extensions import SamplingGridVisualizer
from models import STCNN

img_size = (42, 42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-iter', type=int, default=1000)
    parser.add_argument('--n-sample', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def transform_mnist_rts(in_data):
    img, label = in_data
    img = img[0]  # Remove channel axis for skimage manipulation

    # Rotate
    img = transform.rotate(img, angle=np.random.uniform(-45, 45),
                           resize=True, mode='constant')
    #  Scale
    img = transform.rescale(img, scale=np.random.uniform(0.7, 1.2),
                            mode='constant')

    # Translate
    h, w = img.shape
    if h >= img_size[0] or w >= img_size[1]:
        img = transform.resize(img, output_shape=img_size, mode='constant')
        img = img.astype(np.float32)
    else:
        img_canvas = np.zeros(img_size, dtype=np.float32)
        ymin = np.random.randint(0, img_size[0] - h)
        xmin = np.random.randint(0, img_size[1] - w)
        img_canvas[ymin:ymin+h, xmin:xmin+w] = img
        img = img_canvas

    img = img[np.newaxis, :]  # Add the bach channel back
    return img, label


if __name__ == '__main__':
    args = parse_args()

    train, test = datasets.get_mnist(ndim=3)
    train = TransformDataset(train, transform_mnist_rts)
    train_iter = iterators.SerialIterator(train, batch_size=args.batch_size)

    model = L.Classifier(predictor=STCNN())
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = Trainer(updater=updater, stop_trigger=args.max_iter,
                      out=args.out)
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'main/accuracy', 'elapsed_time']),
        trigger=(1, 'iteration'))
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'main/accuracy'], trigger=(1, 'iteration')))
    x_fixed = np.empty((args.n_sample, 1, *img_size), dtype=np.float32)
    for i in range(args.n_sample):
        x_fixed[i] = train[i][0]
    trainer.extend(SamplingGridVisualizer(x_fixed), trigger=(1, 'iteration'))
    trainer.run()
