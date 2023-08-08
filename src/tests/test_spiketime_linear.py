import unittest
import torch
from torch.testing import assert_close

from src.utils_spiketime_linear import get_spiketime, get_spiketime_derivative


class TestSpiketimeLinear(unittest.TestCase):
    def test_spiketime(self):
        """
        Test that the explicit formula for output spike time is correct, using 2 input spikes and 3 output spikes.
        """
        input_spikes = torch.tensor([[0., 0.1], [0.,0.5]]) # note: they need to be ordered, and the weights sorted accordingly
        input_weights = torch.tensor([
            [[3.,1.,0.], [1.,3.,1.]],
            [[3.,1.,0.], [1.,3.,1.]]
        ]) # n_batch x n_in x n_out
        neuron_params = {
            "threshold": 1.0
        }
        device = "cpu"
        output_spikes_solution = torch.tensor([
            [[1./3., 1., 1e10], [11./40., 13./40., 1.1]],
            [[1./3., 1., 1e10], [3./8., 5./8., 1.5]]
        ]) # n_batch x n_in x n_out 
        # first field: spike times with first input only, second field: with both inputs
        # note: value at (1, 1, 0) consistent with the formula 1/4(1+3*0+1*0.5)=3/8, even though the spike time is 1/3
        # but this is treated correctly in wrapping function
        output_spikes = get_spiketime(input_spikes, input_weights, neuron_params, device)
        assert_close(output_spikes, output_spikes_solution)

    def test_spiketime_derivative(self):
        input_spikes = torch.tensor([[0., 0.1],[0., 0.5]]) # note: they are ordered in the wrapping function, and the weights sorted accordingly
        input_weights = torch.tensor([[[3.,1.,0.], [1.,3.,1.]],[[3.,1.,0.], [1.,3.,1.]]]) # n_batch x n_in x n_out
        neuron_params = {
            "threshold": 1.0
        } 
        device = "cpu"
        output_spikes = torch.tensor([[11./40., 13./40., 1.1], [1./3., 5./8., 1.5]]) # n_batch x n_out (now only the actual spike times, i.e. the first)
        
        dw, dt = get_spiketime_derivative(input_spikes, input_weights, neuron_params, device, output_spikes)
        dw_solution = torch.tensor([
            [[-11./40./4., -13./40./4., -1.1/1.], [(0.1-11./40.)/4., (0.1-13./40.)/4., (0.1-1.1)/1.]],
            [[-1./3./3., -5./8./4., -1.5/1.], [0., (0.5-5./8.)/4., (0.5-1.5)/1.]]
        ]) # n_batch x n_in x n_out
        # formula: dt_v / dw_uv = -(t_v-t_u) / sum of causal weights
        dt_solution = torch.tensor([
            [[3./4., 1./4., 0./1.],[1./4., 3./4., 1./1.]], # need to keep track of how many spikes are causal (ignore the other weights, cf. output spikes)
            [[3./3., 1./4., 0./1.],[0., 3./4., 1./1.]] # 0 if spike arrived too late to affect output spike
        ]) # n_batch x n_in x n_out
        # formula: dt_v/dt_u = w_uv / sum of causal weights
        assert_close(dw, dw_solution)
        assert_close(dt, dt_solution)


if __name__ == '__main__':
    unittest.main()
