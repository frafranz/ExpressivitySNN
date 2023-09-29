import numpy as np
import torch

from utils import to_device
from spiking_layer import SpikingLayer


class SpikingNet(torch.nn.Module):
    """
    Spiking Neural Network model

    Consisting of layers with adjustable sizes, initialization parameters and bias spikes.
    """
    def __init__(self, network_layout, sim_params, device):
        """
        Setup up an SNN

        Arguments:
            network_layout: dict containing the number of inputs, layers, their sizes and initialization parameters
            sim_params: simulation parameters, e.g. for rounding
            device: the device to operate on
        """
        super().__init__()
        self.n_inputs = network_layout['n_inputs']
        self.n_layers = network_layout['n_layers']
        self.layer_sizes = network_layout['layer_sizes']
        self.n_biases = network_layout['n_biases']
        self.weight_means = network_layout['weight_means']
        self.weight_stdevs = network_layout['weight_stdevs']
        
        if network_layout.get('substitute_delays'):
            self.delay_means = network_layout.get('delay_means', [-10.]*self.n_layers)
        else:
            self.delay_means = network_layout.get('delay_means', [0.]*self.n_layers)
        self.delay_stdevs = network_layout.get('delay_stdevs', [0.]*self.n_layers)

        self.threshold_means = network_layout.get('threshold_means', [sim_params['threshold']]*self.n_layers)
        self.threshold_stdevs = network_layout.get('threshold_stdevs', [0.]*self.n_layers)

        self.device = device

        if network_layout.get('bias_times'):
            if len(network_layout['bias_times']) > 0 and isinstance(network_layout['bias_times'][0], (list, np.ndarray)):
                self.bias_times = network_layout['bias_times']
            else:
                self.bias_times = [network_layout['bias_times']] * self.n_layers
        else:
            self.bias_times = []
        self.biases = []
        print(self.bias_times)
        for i in range(self.n_layers):
            bias = to_device(self.bias_inputs(self.n_biases[i], self.bias_times[i]), device)
            self.biases.append(bias)

        self.layers = torch.nn.ModuleList()
        layer = SpikingLayer(self.n_inputs, self.layer_sizes[0],
                             sim_params, (self.weight_means[0], self.weight_stdevs[0]),
                             (self.delay_means[0], self.delay_stdevs[0]),
                             (self.threshold_means[0], self.threshold_stdevs[0]),
                             device, self.n_biases[0])
        self.layers.append(layer)
        for i in range(self.n_layers - 1):
            layer = SpikingLayer(self.layer_sizes[i], self.layer_sizes[i + 1],
                                 sim_params, (self.weight_means[i + 1], self.weight_stdevs[i + 1]),
                                 (self.delay_means[i+1], self.delay_stdevs[i+1]),
                                 (self.threshold_means[i+1], self.threshold_stdevs[i+1]),
                                 device, self.n_biases[i + 1])
            self.layers.append(layer)

        self.rounding_precision = sim_params.get('rounding_precision')
        self.rounding = self.rounding_precision not in (None, False)
        self.sim_params = sim_params

        if self.rounding:
            print(f"#### Rounding the weights to precision {self.rounding_precision}")
        return
    
    def forward(self, input_times):
        """
        Compute the forward pass of the whole network

        Arguments:
            input_times: incoming spike times
        
        Returns:
            label_times, hidden_times: the spike times of the output layer and all hidden layers
        """
        # When rounding we need to save and manipulate weights before forward pass, and after
        if self.rounding:
            float_weights = []
            for layer in self.layers:
                float_weights.append(layer.weight.data)
                layer.weight.data = self.round_weights(layer.weight.data, self.rounding_precision)

        # below, the actual pass through the layers of the network is defined, including the bias terms
        hidden_times = []
        for i in range(self.n_layers):
            input_times_including_bias = torch.cat(
                (input_times,
                    self.biases[i].view(1, -1).expand(len(input_times), -1)),
                1)
            # n_spikes += len(self.biases[i]) leave out number of biases since it is insignificant for the number of spikes
            output_times = self.layers[i](input_times_including_bias)
            if not i == (self.n_layers - 1):
                hidden_times.append(output_times)
                input_times = output_times
            else:
                label_times = output_times
        return_value = label_times, hidden_times

        if self.rounding:
            for layer, floats in zip(self.layers, float_weights):
                layer.weight.data = floats

        return return_value

    def spike_percentages(self, output_times, hidden_times):
        """
        Compute the ratio of neurons that spike during a forward pass.
        
        Parameters:
            output_times: spike times of the final layer
            hidden_times: spikes times in all hidden layers

        Returns:
            one ratio for each layer, e.g. 0.5, (1.0,0.5) for one output layer and two hidden ones
        """
        n_output = torch.isfinite(output_times).sum(dim=1).float().mean() # sum over spikes and average over batches
        n_hidden = [torch.isfinite(hidden).sum(dim=1).float().mean() for hidden in hidden_times]
        output_percentage = n_output / self.layer_sizes[-1]
        hidden_percentages = [n_hidden[i] / self.layer_sizes[-2-i] for i in range(self.n_layers-1)]
        return output_percentage, hidden_percentages
    
    @staticmethod
    def bias_inputs(number_biases, t_bias):
        assert len(t_bias) == number_biases
        times = [t_bias[i] for i in range(number_biases)]
        return torch.tensor(times)
    
    @staticmethod
    def round_weights(weights, precision):
        return (weights / precision).round() * precision

    def clip_weights(self):
        if self.sim_params['clip_weights_max']:
            for i, layer in enumerate(self.layers):
                maxweight = self.sim_params['clip_weights_max']
                self.layers[i].weight.data = torch.clamp(layer.weight.data, -maxweight, maxweight)
        return
    