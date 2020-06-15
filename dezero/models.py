from dezero import Layer
from dezero import utils
import numpy as np
import dezero.functions as F
import dezero.layers as L
from dezero import utils

class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)


class MLP(Layer):
    def __init__(self, fc_output_sizes, activation=F.sigmoid): #fc_output_sizes is tuple or list (output size of the full connected layer)
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l' + str(i), layer) #setattr function works like self.l2 = L.Linear(out_size)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)



