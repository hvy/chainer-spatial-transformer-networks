import chainer
from chainer import functions as F
from chainer import links as L


class STCNN(chainer.Chain):
    def __init__(self, in_channels=1, hidden_channels=32, n_out=10):
        super(STCNN, self).__init__()
        with self.init_scope():
            self.st = ConvAffineSpatialTransformer(in_channels)
            self.conv1 = L.Convolution2D(in_channels, hidden_channels, ksize=9,
                                         stride=1, pad=0)
            self.conv2 = L.Convolution2D(hidden_channels, hidden_channels,
                                         ksize=7, stride=1, pad=0)
            self.fc = L.Linear(n_out)

    def __call__(self, x):
        h = self.st(x)
        h = F.average_pooling_2d(h, 2, 2)  # For TC and RTS datasets
        h = F.relu(self.conv1(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = self.fc(h)
        return h


class ConvAffineSpatialTransformer(chainer.Chain):
    def __init__(self, in_channels=1, hidden_channels=20):
        super(ConvAffineSpatialTransformer, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels, hidden_channels, 5, 1, 0)
            self.conv2 = L.Convolution2D(hidden_channels,
                                         hidden_channels, 5, 1, 0)

            # Initialize the affine matrix with the identity transformation
            identity_bias = self.xp.array([1, 0, 0, 0, 1, 0],
                                          dtype=self.xp.float32)
            self.fc = L.Linear(None, 6,
                               initialW=chainer.initializers.Constant(0),
                               initial_bias=identity_bias)

    def affine_matrix(self, x):
        h = F.max_pooling_2d(x, 2, 2)
        h = F.relu(self.conv1(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2, 2)
        theta = F.reshape(self.fc(h), (x.shape[0], 2, 3))
        return theta

    def __call__(self, x):
        theta = self.affine_matrix(x)
        self.grid = F.spatial_transformer_grid(theta, x.shape[2:])
        return F.spatial_transformer_sampler(x, self.grid)
