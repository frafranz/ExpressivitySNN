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

    Called from EqualtimeFunction below, for each layer.
    Dimensions are crucial:
        input spikes have dimension batchesxN with N presynaptic neurons
        input weights have dimension BATCHESxNxM with N pre and M postsynaptic neurons.
        thresholds have dimension M, corresponding to all postsynaptic neurons
    """
    n_batch, n_presyn = input_spikes.shape
    n_batch2, n_presyn2, n_postsyn = input_weights.shape
    n_postsyn2 = thresholds.shape[0]
    assert n_batch == n_batch2, "Deep problem with unequal batch sizes"
    assert n_presyn == n_presyn2
    assert n_postsyn == n_postsyn2

    tau_syn = neuron_params['tau_syn']
    tau_mem = neuron_params['tau_mem']
    assert tau_syn / tau_mem == neuron_params['g_leak'], \
        f"did you set g_leak according to tau ratio (probably {tau_syn / tau_mem}, " \
        f"currently {neuron_params['g_leak']})"

    exponentiated_spike_times_syn = torch.exp(input_spikes / tau_syn)
    exponentiated_spike_times_mem = torch.exp(input_spikes / tau_mem)

    # to prevent NaNs when first (sorted) weight(s) is 0, thus A and B, and ratio NaN add epsilon
    eps = 1e-6
    factor_a1 = torch.cumsum(exponentiated_spike_times_syn.unsqueeze(-1) * input_weights, dim=1) + eps
    factor_a2 = torch.cumsum(exponentiated_spike_times_mem.unsqueeze(-1) * input_weights, dim=1) + eps
    factor_c = (thresholds.view(1, 1, n_postsyn) - neuron_params['leak']) * neuron_params['g_leak']

    factor_sqrt = torch.sqrt(factor_a2 ** 2 - 4 * factor_a1 * factor_c)

    ret_val = 2. * tau_syn * torch.log(2. * factor_a1 / (factor_a2 + factor_sqrt))
    ret_val[torch.isnan(ret_val)] = float('inf')
    
    return ret_val


def get_spiketime_derivative(input_spikes, input_weights, neuron_params, device,
                             output_spikes, thresholds):
    """Calculating the derivatives, see above.

    Input spikes have shape batch,presyn, weights have shape batch,presyn,postsyn, and both are ordered according to input spike times.

    """
    n_batch, n_presyn = input_spikes.shape
    n_batch2, n_presyn2, n_postsyn = input_weights.shape
    n_postsyn2 = thresholds.shape[0]
    assert n_batch == n_batch2, "Deep problem with unequal batch sizes"
    assert n_presyn == n_presyn2
    assert n_postsyn == n_postsyn2

    output_minus_input = -input_spikes.view(n_batch, n_presyn, 1) + output_spikes.view(n_batch, 1, n_postsyn)
    mask = (output_minus_input < 0) | torch.isinf(output_minus_input) | torch.isnan(output_minus_input)
    causal_weights = input_weights
    # set infinities to 0 preventing nans
    causal_weights[mask] = 0.
    input_spikes[torch.isinf(input_spikes)] = 0.
    output_spikes[torch.isinf(output_spikes)] = 0.

    input_spikes = input_spikes.view(n_batch, 1, n_presyn)

    tau_syn = neuron_params['tau_syn']
    tau_mem = neuron_params['tau_mem']
    exponentiated_spike_times_syn = torch.exp(input_spikes / tau_syn)
    exponentiated_spike_times_mem = torch.exp(input_spikes / tau_mem)
    exponentiated_spike_times_out_mem = torch.exp(output_spikes / tau_mem)

    eps = 1e-6
    factor_a1 = torch.matmul(exponentiated_spike_times_syn, causal_weights) + eps
    factor_a2 = torch.matmul(exponentiated_spike_times_mem, causal_weights) + eps
    factor_c = (thresholds.view(1, 1, n_postsyn) - neuron_params['leak']) * neuron_params['g_leak']

    factor_sqrt = torch.sqrt(factor_a2 ** 2 - 4 * factor_a1 * factor_c)

    exponentiated_spike_times_out_mem = exponentiated_spike_times_out_mem.squeeze().unsqueeze(1)
    exponentiated_spike_times_syn = exponentiated_spike_times_syn.squeeze().unsqueeze(-1)
    exponentiated_spike_times_mem = exponentiated_spike_times_mem.squeeze().unsqueeze(-1)

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

    return dw, dt, dtheta
