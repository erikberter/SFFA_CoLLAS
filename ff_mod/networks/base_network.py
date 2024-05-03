import torch
from torch.optim import Adam

from torch.nn import functional as F

from ff_mod.networks.abstract_forward_forward import AbstractForwardForwardNet, AbstractForwardForwardLayer

from ff_mod.overlay import Overlay

class AnalogNetwork(AbstractForwardForwardNet):

    def __init__(
            self,
            overlay_function : Overlay,
            dims,
            loss_function,
            learning_rate = 0.001,
            internal_epoch = 20,
            residual_connections = True,
            normalize_activity = True,
            first_prediction_layer = 0
        ):
        
        super().__init__(overlay_function, first_prediction_layer, False)
        
        self.internal_epoch = internal_epoch
        
        self.residual_connections = residual_connections
        
        for d in range(len(dims) - 1):
            layer = Layer(dims[d], dims[d + 1], loss_function, learning_rate, internal_epoch, normalize_activity=normalize_activity)
            self.layers.append(layer)

        self.layers[0].is_input_layer = True
        
    def adjust_data(self, data):
        return data

class Layer(AbstractForwardForwardLayer):
    def __init__(
            self,
            in_features,
            out_features,
            loss_function,
            learning_rate = 0.001,
            internal_epoch = 10,
            normalize_activity = False,
            **kwargs
        ):
        super().__init__(in_features, out_features, loss_function = loss_function, **kwargs)
        
        self.is_input_layer = False
        
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(in_features)
        
        self.opt = Adam(self.parameters(), lr=learning_rate)

        self.num_epochs = internal_epoch
        
        self.normalize_activity = normalize_activity

    def get_goodness(self, x):
        
        goodness = x.pow(2).mean(1)
        
        print(f"Goodness mean {goodness.mean()}")
        goodness = F.sigmoid(goodness - 2)
        
        
        goodness = torch.clamp(goodness, 0+0.0001, 1-0.0001)
        
        return goodness
    
    def forward(self, x):
        
        bias_term = 0
        if self.bias is not None:
            bias_term = self.bias.unsqueeze(0)
        
        if not self.is_input_layer and self.normalize_activity:
            x = x / (x.pow(2).mean(1).sqrt().unsqueeze(1) + 1e-10)

        result =  self.relu(torch.mm(x, self.weight.T) + bias_term)
        
        return result
