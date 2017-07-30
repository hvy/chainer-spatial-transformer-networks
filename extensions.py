import matplotlib  # NOQA
matplotlib.use('Agg')  # NOQA

import math
import os

from chainer import cuda
from chainer.training import extension
import matplotlib.pyplot as plt
import numpy as np


class SamplingGridVisualizer(extension.Extension):
    def __init__(self, x, dpi=100):
        self.x = x
        self.dpi = dpi

    def __call__(self, trainer):
        x = self.x
        dpi = self.dpi
        updater = trainer.updater

        filename = os.path.join(trainer.out, '{0:08d}.png'.format(
                                updater.iteration))

        # Inference to update model internal grid
        x = updater.converter(x, updater.device)
        model = updater.get_optimizer('main').target.predictor
        model(x)

        # Get grids from previous inference
        grid = model.st.grid.data
        if isinstance(grid, cuda.ndarray):
            grid = cuda.to_cpu(grid)
        if isinstance(x, cuda.ndarray):
            x = cuda.to_cpu(x)

        n, c, w, h = x.shape
        x_plots = math.ceil(math.sqrt(n))
        y_plots = x_plots if n % x_plots == 0 else x_plots - 1
        plt.figure(figsize=(w*x_plots/dpi, h*y_plots/dpi), dpi=dpi)

        for i, im in enumerate(x):
            plt.subplot(y_plots, x_plots, i+1)

            if c == 1:
                plt.imshow(im[0])
            else:
                plt.imshow(im.transpose((1, 2, 0)))

            plt.axis('off')
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.gray()

            # Get the 4 corners of the transformation grid to draw a box
            g = grid[i]
            vs = np.empty((4, 2), dtype=np.float32)
            vs[0] = g[:, 0, 0]
            vs[1] = g[:, 0, w-1]
            vs[2] = g[:, h-1, w-1]
            vs[3] = g[:, h-1, 0]
            vs += 1  # [-1, 1] -> [0, 2]
            vs /= 2
            vs[:, 0] *= h
            vs[:, 1] *= w

            bbox = plt.Polygon(vs, True, color='r', fill=False, linewidth=0.8,
                               alpha=0.8)
            plt.gca().add_patch(bbox)
            bbox.set_clip_on(False)  # Allow drawing outside axes

            plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                wspace=0.2, hspace=0.2)

        plt.savefig(filename, dpi=dpi*2, facecolor='black')
        plt.clf()
        plt.close()
