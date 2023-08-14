import unittest
import torch
from torch.testing import assert_close

from utils_spiketime_linear import get_spiketime, get_spiketime_derivative


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
        input_delays = torch.zeros_like(input_weights)
        delayed_input_spikes = input_spikes.unsqueeze(-1) + input_delays
        neuron_params = {
            "threshold": 1.0
        }
        device = "cpu"
        output_spikes_solution = torch.tensor([
            [[(1+3*0)/3., (1+1*0)/1., torch.inf], [(1+3*0+1*0.1)/4., (1+1*0+3*0.1)/4., (1+1*0.1)/1.]],
            [[(1+3*0)/3., (1+1*0)/1., torch.inf], [(1+3*0+1*0.5)/4., (1+1*0+3*0.5)/4., (1+0*0+1*0.5)/1.]]
        ]) # n_batch x n_in x n_out 
        # first field: spike times with first input only, second field: with both inputs
        # note: value at (1, 1, 0) consistent with the formula 1/4(1+3*0+1*0.5)=3/8, even though the spike time is 1/3
        # but this is treated correctly in wrapping function
        output_spikes = get_spiketime(delayed_input_spikes, input_weights, neuron_params, device)
        assert_close(output_spikes, output_spikes_solution)

    def test_spiketime_with_delay(self):
        """
        Test that the explicit formula for output spike time remains correct if delays are introduced.
        """
        input_spikes = torch.tensor([[0., 0.1], [0.,0.5]]) # note: they need to be ordered, and the weights sorted accordingly
        input_weights = torch.tensor([
            [[3.,1.,0.], [1.,3.,1.]],
            [[3.,1.,0.], [1.,3.,1.]]
        ]) # n_batch x n_in x n_out
        input_delays = torch.tensor([
            [[0.1,0.05,0.3], [.01,.2,0.5]],
            [[0.1,0.05,0.3], [.01,.2,0.5]]
        ]) # n_batch x n_in x n_out
        delayed_input_spikes = input_spikes.unsqueeze(-1) + input_delays
        neuron_params = {
            "threshold": 1.0
        }
        device = "cpu"
        output_spikes_solution = torch.tensor([
            [[(1+3*(0+0.1))/3., (1+1*(0+0.05))/1., torch.inf], [(1+3*(0+0.1)+1*(0.1+0.01))/4., (1+1*(0+0.05)+3*(0.1+0.2))/4., (1+0*(0+0.3)+1*(0.1+0.5))/1.]],
            [[(1+3*(0+0.1))/3., (1+1*(0+0.05))/1., torch.inf], [(1+3*(0+0.1)+1*(0.5+0.01))/4., (1+1*(0+0.05)+3*(0.5+0.2))/4., (1+0*(0+0.3)+1*(0.5+0.5))/1.]]
        ]) # n_batch x n_in x n_out 
        # for each batch, first field: spike times with first input only, second field: with both inputs
        # note: value at (1, 1, 0) consistent with the formula 1/4(1+3*0+1*0.5)=3/8, even though the spike time is 1/3
        # but this is treated correctly in wrapping function
        output_spikes = get_spiketime(delayed_input_spikes, input_weights, neuron_params, device)
        assert_close(output_spikes, output_spikes_solution)

    def test_spiketime_derivative(self):
        """
        Test that the derivatives of the output spike times are correct, with respect to input weights and input spike times. 
        The same small SNN is used as in the previous tests. The training of delays is explicitly turned off here.
        """
        input_spikes = torch.tensor([[0., 0.1],[0., 0.5]]) # note: they are ordered in the wrapping function, and the weights sorted accordingly
        input_weights = torch.tensor([[[3.,1.,0.], [1.,3.,1.]],
                                      [[3.,1.,0.], [1.,3.,1.]]]) # n_batch x n_in x n_out
        input_delays = torch.zeros_like(input_weights)
        delayed_input_spikes = input_spikes.unsqueeze(-1) + input_delays
        neuron_params = {
            "threshold": 1.0,
            "train_delay": False
        } 
        device = "cpu"
        output_spikes = torch.tensor([[11./40., 13./40., 1.1], [1./3., 5./8., 3./2.]]) # n_batch x n_out (now only the actual spike times)
        
        dw, dt, dd = get_spiketime_derivative(delayed_input_spikes, input_weights, neuron_params, device, output_spikes)
        dw_solution = torch.tensor([
            [[-output_spikes[0,0]/4., -output_spikes[0,1]/4., -output_spikes[0,2]/1.], 
             [(0.1-output_spikes[0,0])/4., (0.1-output_spikes[0,1])/4., (0.1-output_spikes[0,2])/1.]],
            [[-output_spikes[1,0]/3., -output_spikes[1,1]/4., -output_spikes[1,2]/1.], 
             [0./3., (0.5-output_spikes[1,1])/4., (0.5-output_spikes[1,2])/1.]]
        ]) # n_batch x n_in x n_out
        # formula: dt_v / dw_uv = (t_u-t_v) / sum of causal weights
        dt_solution = torch.tensor([
            [[3./4., 1./4., 0./1.],[1./4., 3./4., 1./1.]], # need to keep track of how many spikes are causal (ignore the other weights, cf. output spikes)
            [[3./3., 1./4., 0./1.],[0., 3./4., 1./1.]] # 0 if spike arrived too late to affect output spike
        ]) # n_batch x n_in x n_out
        # formula: dt_v/dt_u = w_uv / sum of causal weights
        dd_solution = torch.zeros_like(dt_solution)
        assert_close(dw, dw_solution)
        assert_close(dt, dt_solution)
        assert_close(dd, dd_solution)

    def test_spiketime_derivative_with_zero_delay(self):
        """
        Test that the derivatives of the output spike times remain correct when delays between neurons are introduced.
        Here we consider the simple case that all delays are zero.
        """
        input_spikes = torch.tensor([[0., 0.1],[0., 0.5]]) # note: they are ordered in the wrapping function, and the weights sorted accordingly
        input_weights = torch.tensor([[[3.,1.,0.], [1.,3.,1.]],[[3.,1.,0.], [1.,3.,1.]]]) # n_batch x n_in x n_out
        input_delays = torch.zeros_like(input_weights)
        delayed_input_spikes = input_spikes.unsqueeze(-1) + input_delays
        neuron_params = {
            "threshold": 1.0,
            "train_delay": True
        }
        device = "cpu"
        output_spikes = torch.tensor([[11./40., 13./40., 1.1], [1./3., 5./8., 3./2.]]) # n_batch x n_out (now only the actual spike times)

        dw, dt, dd = get_spiketime_derivative(delayed_input_spikes, input_weights, neuron_params, device, output_spikes)
        dw_solution = torch.tensor([
            [[-output_spikes[0,0]/4., -output_spikes[0,1]/4., -output_spikes[0,2]/1.], 
             [(0.1-output_spikes[0,0])/4., (0.1-output_spikes[0,1])/4., (0.1-output_spikes[0,2])/1.]],
            [[-output_spikes[1,0]/3., -output_spikes[1,1]/4., -output_spikes[1,2]/1.], 
             [0./3., (0.5-output_spikes[1,1])/4., (0.5-output_spikes[1,2])/1.]]
        ]) # n_batch x n_in x n_out
        # formula: dt_v / dw_uv = (t_u-t_v) / sum of causal weights
        dt_solution = torch.tensor([
            [[3./4., 1./4., 0./1.],[1./4., 3./4., 1./1.]], # need to keep track of how many spikes are causal (ignore the other weights, cf. output spikes)
            [[3./3., 1./4., 0./1.],[0., 3./4., 1./1.]] # 0 if spike arrived too late to affect output spike
        ]) # n_batch x n_in x n_out
        # formula: dt_v/dt_u = w_uv / sum of causal weights
        dd_solution = dt_solution.clone()
        assert_close(dw, dw_solution)
        assert_close(dt, dt_solution)
        assert_close(dd, dd_solution)

    def test_spiketime_derivative_with_delay(self):
        """
        Test that the derivatives of the output spike times remain correct when (nontrivial) delays between neurons are introduced.
        """
        input_spikes = torch.tensor([[0., 0.1],[0., 0.5]]) # note: they are ordered in the wrapping function, and the weights sorted accordingly
        input_weights = torch.tensor([[[3.,1.,0.], [1.,3.,1.]],[[3.,1.,0.], [1.,3.,1.]]]) # n_batch x n_in x n_out
        input_delays = torch.tensor([[[0.1,0.05,0.3], [.01,.2,0.5]],
                                     [[0.1,0.05,0.3], [.01,.2,0.5]]]) # n_batch x n_in x n_out
        delayed_input_spikes = input_spikes.unsqueeze(-1) + input_delays
        neuron_params = {
            "threshold": 1.0,
            "train_delay": True
        } 
        device = "cpu"
        output_spikes = torch.tensor([
            [(1+3*(0+0.1)+1*(0.1+0.01))/4., (1+1*(0+0.05)+3*(0.1+0.2))/4., (1+0*(0+0.3)+1*(0.1+0.5))/1.],
            [(1+3*(0+0.1))/3., (1+1*(0+0.05)+3*(0.5+0.2))/4., (1+0*(0+0.3)+1*(0.5+0.5))/1.]
        ]) # n_batch x n_out

        dw, dt, dd = get_spiketime_derivative(delayed_input_spikes, input_weights, neuron_params, device, output_spikes)
        dw_solution = torch.tensor([
            [[(0.1-output_spikes[0,0])/4., (0.05-output_spikes[0,1])/4., (0.3-output_spikes[0,2])/1.], 
             [(0.1+0.01-output_spikes[0,0])/4., (0.1+0.2-output_spikes[0,1])/4., (0.1+0.5-output_spikes[0,2])/1.]],
            [[(0.1-output_spikes[1,0])/3., (0.05-output_spikes[1,1])/4., (0.3-output_spikes[1,2])/1.], 
             [0./3., (0.5+0.2-output_spikes[1,1])/4., (0.5+0.5-output_spikes[1,2])/1.]]
        ]) # n_batch x n_in x n_out
        # formula: dt_v / dw_uv = -(t_v-t_u) / sum of causal weights
        dt_solution = torch.tensor([
            [[3./4., 1./4., 0./1.],[1./4., 3./4., 1./1.]],
            [[3./3., 1./4., 0./1.],[0, 3./4., 1./1.]] # 0 if spike arrived too late to affect output spike
        ]) # n_batch x n_in x n_out
        # formula: dt_v/dt_u = w_uv / sum of causal weights
        dd_solution = dt_solution.clone()
        assert_close(dw, dw_solution)
        assert_close(dt, dt_solution)
        assert_close(dd, dd_solution)


if __name__ == '__main__':
    unittest.main()
