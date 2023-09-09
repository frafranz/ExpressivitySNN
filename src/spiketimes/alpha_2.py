#!python3
"""this file contains the update rule for tau_mem = 2 * tau_syn
one can exchange this file in utils.py for the other, but the
parameters have to be adapted as well.
"""
import numba
import numpy as np
import os
import os.path as osp
import sys
import time
import torch
import torch.nn
import torch.autograd
import yaml


def get_spiketime(input_spikes, input_weights, thresholds, neuron_params, device):
    """Calculating spike times, all at once.

    Called from the explicit spiketime simulation, for each layer. The function relies on the incoming spike times with their respective weights being sorted.

    Parameters:
        (delayed) input spikes, input weights: have dimension BATCHESxNxM with N pre and M postsynaptic neurons, 
        where the last coordinate represents the output neuron for which the input is considered
        thresholds: have dimension M, corresponding to all postsynaptic neurons
        neuron_params: parameters of the spiking neuron
        device: device to operate on
    
    
    Returns:
        Of shape (n_batch, n_presyn, n_postsyn), contains the time of on outgoing spike to neuron n_postsyn (for n_batch),
        given that all input spikes up to the one at n_presyn have arrived.
    """
    n_batch, n_presyn, n_postsyn = input_spikes.shape
    n_batch2, n_presyn2, n_postsyn2 = input_weights.shape
    n_postsyn3 = thresholds.shape[0]
    assert n_batch == n_batch2, "Deep problem with unequal batch sizes"
    assert n_presyn == n_presyn2
    assert n_postsyn == n_postsyn2 == n_postsyn3

    # fastest implementation: multiply spike times by weights and finally sum over causal spike times
    tau_syn = neuron_params['tau_syn']
    tau_mem = neuron_params['tau_mem']
    assert tau_syn / tau_mem == neuron_params['g_leak'], \
        f"did you set g_leak according to tau ratio (probably {tau_syn / tau_mem}, " \
        f"currently {neuron_params['g_leak']})"

    exponentiated_spike_times_syn = torch.exp(input_spikes / tau_syn)
    exponentiated_spike_times_mem = torch.exp(input_spikes / tau_mem)

    # to prevent NaNs when first (sorted) weight(s) is 0, thus A and B, and ratio NaN add epsilon
    eps = 1e-6
    factor_a1 = torch.cumsum(exponentiated_spike_times_syn*input_weights, dim=1) + eps # n_batch x n_presyn x postsyn
    factor_a2 = torch.cumsum(exponentiated_spike_times_mem*input_weights, dim=1) + eps # also added eps here
    factor_c = (thresholds.view(1, 1, n_postsyn) - neuron_params['leak']) * neuron_params['g_leak']

    factor_sqrt = torch.sqrt(factor_a2 ** 2 - 4 * factor_a1 * factor_c)

    ret_val = 2. * tau_syn * torch.log(2. * factor_a1 / (factor_a2 + factor_sqrt))
    ret_val[torch.isnan(ret_val)] = float('inf')
    
    return ret_val


def get_spiketime_derivative(input_spikes, input_weights, neuron_params, device,
                             output_spikes, input_delays, thresholds):
    """Calculating the derivatives, see above.

    Weights and (delayed) input spikes have shape batch,presyn,postsyn, are ordered according to spike times.
    """
    n_batch, n_presyn, n_postsyn = input_spikes.shape
    n_batch2, n_presyn2, n_postsyn2 = input_weights.shape
    n_batch3, n_postsyn3 = output_spikes.shape
    n_presyn3, n_postsyn4 = input_delays.shape
    assert n_batch == n_batch2 == n_batch3, "Deep problem with unequal batch sizes"
    assert n_presyn == n_presyn2 == n_presyn3
    assert n_postsyn == n_postsyn2 == n_postsyn3 == n_postsyn4

    output_minus_input = -input_spikes + output_spikes.view(n_batch, 1, n_postsyn)
    mask = (output_minus_input < 0) | torch.isinf(output_minus_input) | torch.isnan(output_minus_input)
    causal_weights = input_weights
    # set infinities to 0 preventing nans
    causal_weights[mask] = 0.
    input_spikes[torch.isinf(input_spikes)] = 0.
    output_spikes[torch.isinf(output_spikes)] = 0.

    tau_syn = neuron_params['tau_syn']
    tau_mem = neuron_params['tau_mem']
    exponentiated_spike_times_syn = torch.exp(input_spikes / tau_syn)
    exponentiated_spike_times_mem = torch.exp(input_spikes / tau_mem)
    exponentiated_spike_times_out_mem = torch.exp(output_spikes / tau_mem)

    eps = 1e-6
    factor_a1 = torch.sum(exponentiated_spike_times_syn * causal_weights, dim=1, keepdim=True) + eps
    factor_a2 = torch.sum(exponentiated_spike_times_mem * causal_weights, dim=1, keepdim=True) + eps
    factor_c = (thresholds.view(1, 1, n_postsyn) - neuron_params['leak']) * neuron_params['g_leak']

    factor_sqrt = torch.sqrt(factor_a2 ** 2 - 4 * factor_a1 * factor_c)

    dw = tau_mem * (((1. + factor_c / factor_sqrt * exponentiated_spike_times_out_mem)
                     / factor_a1 * exponentiated_spike_times_syn)
                    - 1. / factor_sqrt * exponentiated_spike_times_mem)
    dt = causal_weights * (((1. + factor_c / factor_sqrt * exponentiated_spike_times_out_mem)
                            / factor_a1 * 2. * exponentiated_spike_times_syn)
                           - 1. / factor_sqrt * exponentiated_spike_times_mem)

    # manually set the uncausal and inf output spike entries 0
    dw[mask] = 0.
    dt[mask] = 0.
    dtheta = torch.zeros_like(dt)
    dd = torch.zeros_like(dt)

    return dw, dt, dtheta, dd
