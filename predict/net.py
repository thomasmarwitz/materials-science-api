from torch import nn


class Network(nn.Module):
    def __init__(self, layer_dims: list):
        """
        Fully Connected layers
        """
        super(Network, self).__init__()

        layers = []
        for in_, out_ in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_, out_))
            layers.append(nn.BatchNorm1d(out_))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.1))  # TODO: try out 0.3

        layers.pop()  # remove last dropout layer
        layers.pop()  # remove last relu layer
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Pass throught network
        """
        res = self.net(x)

        return res
