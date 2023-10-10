import numpy as np
import torch

from utils import to_device
from spiking_layer import SpikingLayer


class ReLUNet(torch.nn.Module):
    """
    Artificial Neural Network model

    Using fully connected feedforward layers and ReLU non-linearities.
    Consisting of layers with adjustable sizes and bias.
    """
    def __init__(self, network_layout, sim_params):
        """
        Setup up an SNN

        Arguments:
            network_layout: dict containing the number of inputs, layers, their sizes and initialization parameters
            sim_params: simulation parameters, e.g. for rounding
        """
        super().__init__()
        self.n_inputs = network_layout['n_inputs']
        self.n_layers = network_layout['n_layers']
        self.layer_sizes = network_layout['layer_sizes']
        self.bias = network_layout['bias']
        self.relu = torch.nn.ReLU()
        
        if network_layout.get('input_interval'):
            self.input_interval = network_layout['input_interval']

        self.layers = torch.nn.ModuleList()
        first_layer = torch.nn.Linear(self.n_inputs, self.layer_sizes[0], bias=self.bias[0])
        self.layers.append(first_layer)
        for i in range(self.n_layers - 1):
            layer = torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1], bias=self.bias[i+1])
            self.layers.append(layer)
        
        self.init = network_layout.get('initialization')
        for layer in self.layers:
            if self.init=='He':
                torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            elif self.init=='Xavier':
                torch.nn.init.xavier_uniform_(layer.weight)

        self.rounding_precision = sim_params.get('rounding_precision')
        self.rounding = self.rounding_precision not in (None, False)
        self.sim_params = sim_params

        if self.rounding:
            print(f"#### Rounding the weights to precision {self.rounding_precision}")
        return
    
    def forward(self, inputs):
        """
        Compute the forward pass of the whole network

        Arguments:
            inputs: input data
        
        Returns:
            values: the values within output layer and all hidden layers
            activations: whether values are non-zero
        """
        # When rounding we need to save and manipulate weights before forward pass, and after
        if self.rounding:
            float_weights = []
            for layer in self.layers:
                float_weights.append(layer.weight.data)
                layer.weights.data = self.round_weights(layer.weights.data, self.rounding_precision)

        # below, the actual pass through the layers of the network is defined, including the bias terms
        hidden_values = []
        hidden_activations = []
        for i in range(self.n_layers):
            output_values = self.relu(self.layers[i](inputs))
            if not i == (self.n_layers - 1):
                hidden_values.append(output_values)
                hidden_activations.append(output_values > 0)
                inputs = output_values
            else:
                label_values = output_values
                label_activations = output_values > 0
        values = label_values, hidden_values
        activations = label_activations, hidden_activations

        if self.rounding:
            for layer, floats in zip(self.layers, float_weights):
                layer.weight.data = floats

        return values, activations
    
    @staticmethod
    def round_weights(weights, precision):
        return (weights / precision).round() * precision

    def clip_weights(self):
        if self.sim_params['clip_weights_max']:
            for i, layer in enumerate(self.layers):
                maxweight = self.sim_params['clip_weights_max']
                self.layers[i].weight.data = torch.clamp(layer.weights.data, -maxweight, maxweight)
        return
