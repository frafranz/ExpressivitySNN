import unittest
import torch
from torch.testing import assert_close

from utils import EqualtimeLayer


class TestEqualtimeLayer(unittest.TestCase):
    def test_init(self):
        """
        Test the initialization of one layer of the SNN, with randomized weights and delays.
        No bias input, i.e. an additional fixed-time input spike, is used here.
        """
        input_features = 2
        output_features = 3
        sim_params = {'use_forward_integrator': False, 'activation': 'linear'}
        weights_init = (2, 1) # this uses random weight initialization
        delays_init = (0, 0.1) # also randomized delays
        device = "cpu"
        bias = 0 # number of bias inputs
        layer = EqualtimeLayer(input_features, output_features, sim_params, weights_init, delays_init, device, bias)

    def test_forward(self):
        """
        Test the forward propagation through one layer of the SNN.
        No bias input, i.e. an additional fixed-time input spike, is used here.
        """
        input_features = 2
        output_features = 3
        sim_params = {'use_forward_integrator': False,'threshold': 1.0, 'activation': 'linear'}
        weights_init = torch.tensor([[1.,3.,1.], [3.,1.,0.]]) # n_in x n_out
        delays_init = torch.zeros_like(weights_init)
        device = "cpu"
        bias = 0 # number of bias inputs
        layer = EqualtimeLayer(input_features, output_features, sim_params, weights_init, delays_init, device, bias)
        input_spikes = torch.tensor([[0.1, 0], [0.5,0]]) # n_batch x n_inputs (at this step, the inputs are not yet ordered)
        device = "cpu"
        output_spikes = layer(input_spikes)
        output_spikes_solution = torch.tensor([[11./40., 13./40., 1.1], [1./3., 5./8., 1.5]]) # n_batch x n_out
        assert_close(output_spikes,output_spikes_solution)

    def test_forward_with_delays(self):
        """
        Test the forward propagation through one layer of the SNN, including delays.
        No bias input, i.e. an additional fixed-time input spike, is used here.
        """
        input_features = 2
        output_features = 3
        sim_params = {'use_forward_integrator': False,'threshold': 1.0, 'activation': 'linear'}
        weights_init = torch.tensor([[1.,3.,1.], [3.,1.,0.]]) # n_in x n_out
        delays_init = torch.tensor([[.01,.2,0.5],[0.1,0.05,0.3]]) # n_in x n_out
        device = "cpu"
        bias = 0 # number of bias inputs
        layer = EqualtimeLayer(input_features, output_features, sim_params, weights_init, delays_init, device, bias)
        input_spikes = torch.tensor([[0.1, 0], [0.5,0]]) # n_batch x n_inputs (at this step, the inputs are not yet ordered)
        device = "cpu"
        output_spikes = layer(input_spikes)
        output_spikes_solution = torch.tensor([
            [(1+3*(0+0.1)+1*(0.1+0.01))/4., (1+1*(0+0.05)+3*(0.1+0.2))/4., (1+0*(0+0.3)+1*(0.1+0.5))/1.],
            [(1+3*(0+0.1))/3., (1+1*(0+0.05)+3*(0.5+0.2))/4., (1+0*(0+0.3)+1*(0.5+0.5))/1.]
        ])
        assert_close(output_spikes,output_spikes_solution)

    def test_backward(self):
        # TODO
        pass


if __name__ == '__main__':
    unittest.main()
