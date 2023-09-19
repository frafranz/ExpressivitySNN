import numpy as np
import torch

from simulations.explicit import Explicit
from simulations.time_discretized import TimeDiscretized


class SpikingLayer(torch.nn.Module):
    """
    One layer of the SNN
    """
    def __init__(self, input_features, output_features, sim_params, weights_init, delays_init, thresholds_init,
                 device, bias=0):
        """Setup up a layer of spiking neurons

        Arguments:
            input_features, output_features: number of inputs/outputs
            sim_params: parameters used for simulation
            weights_init: if tuple it is understood as two lists of mean and std, otherwise an array of weights
            delays_init: if tuple it is understood as two lists of mean and std, otherwise an array of weights
            thresholds_init: if tuple it is understood as two lists of mean and std, otherwise an array of weights
            device: the device to operate on
            bias: number of bias inputs
        """
        super(SpikingLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.sim_params = sim_params
        self.bias = bias
        self.device = device
        self.use_forward_integrator = sim_params.get('use_forward_integrator', False)
        if self.use_forward_integrator:
            assert 'resolution' in sim_params and 'sim_time' in sim_params
            self.sim_params['steps'] = int(np.ceil(sim_params['sim_time'] / sim_params['resolution']))
            self.sim_params['decay_syn'] = float(np.exp(-sim_params['resolution'] / sim_params['tau_syn']))
            self.sim_params['decay_mem'] = float(np.exp(-sim_params['resolution'] / sim_params['tau_syn']))
            if self.sim_params.get('delta'):
                self.sim_params['delta_in_steps'] = int(np.ceil(sim_params['delta'] / sim_params['resolution']))
                
        self.weights = torch.nn.Parameter(torch.Tensor(input_features + bias, output_features))
        self.delays = torch.nn.Parameter(torch.Tensor(input_features + bias, output_features))
        self.thresholds = torch.nn.Parameter(torch.Tensor(output_features))

        if isinstance(weights_init, tuple):
            self.weights.data.normal_(weights_init[0], weights_init[1])
        else:
            assert weights_init.shape == (input_features + bias, output_features)
            self.weights.data = weights_init
        
        if isinstance(delays_init, tuple):
            if self.sim_params.get('substitute_delay'):
                self.delays.data.normal_(delays_init[0], delays_init[1])
            else:
                self.delays.data.normal_(delays_init[0], delays_init[1])
        else:
            assert delays_init.shape == (input_features + bias, output_features)
            self.delays.data = delays_init

        if isinstance(thresholds_init, tuple):
            self.thresholds.data.normal_(thresholds_init[0], thresholds_init[1])
        else:
            assert thresholds_init.shape[0] == output_features
            self.thresholds.data = thresholds_init
        

    def forward(self, input_times):
        """
        Compute the forward pass through one layer.

        Depending on the, configuration use either explicit or time-discretized simulation.

        Parameters:
            input_times: incoming spike times

        Returns:
            the outgoing spike times from this layer
        """
        if self.use_forward_integrator:
            return TimeDiscretized.apply(input_times, self.weights, self.delays, self.thresholds,
                                                     self.sim_params,
                                                     self.device)
        else:
            return Explicit.apply(input_times, self.weights, self.delays, self.thresholds,
                                                     self.sim_params,
                                                     self.device)