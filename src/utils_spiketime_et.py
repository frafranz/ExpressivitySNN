#!python3
import numpy as np
import os
import torch
import torch.nn
import torch.autograd

import utils

torch.set_default_dtype(torch.float64)

if "USE_LAMBERTW_SCIPY" in os.environ or not torch.cuda.is_available():
    from scipy.special import lambertw as lambertw_scipy

    def lambertw(inpt, device):
        # return reals, and set those with nonvanishing imaginary part to -inf
        factorW = lambertw_scipy(inpt.cpu().detach().numpy())
        factorW[np.imag(factorW) != 0] = -np.inf
        factorW = utils.to_device(torch.tensor(np.real(factorW)), device)
        return factorW
else:
    # try to import the module
    try:
        from lambertw_cuda import lambertw0 as lambertw_cuda

        def lambertw(inpt, device):
            ret_val = lambertw_cuda(inpt)
            # cuda lambertw can't return inf and returns 697.789 instead
            ret_val[ret_val > 690.] = float('inf')
            return ret_val

    except ImportError:
        raise NotImplementedError(
            "If you have a GPU, "
            "please go into ./pytorch_cuda_lambertw and "
            "run 'python setup.py install --user'. "
            "Alternatively define USE_LAMBERTW_SCIPY in your env.")
    # test it
    try:
        test_cuda_tensor = torch.ones(1).to(torch.device('cuda'))
        lambertw(test_cuda_tensor, torch.device('cuda'))
    except Exception:
        print("when trying to evalutate the cuda lambertw there was a problem")
        raise


def get_spiketime(input_spikes, input_weights, thresholds, neuron_params, device):
    """Calculating spike times, all at once.

    Called from EqualtimeFunction below, for each layer.
    Dimensions are crucial:
        input weights and (delayed) input spikes have dimension BATCHESxNxM with N pre and M postsynaptic neurons, 
        where the last coordinate represents the output neuron for which the input is considered
        thresholds have dimension M, corresponding to all postsynaptic neurons
    The functions relies on the incoming spike time with their respective weights being sorted.

    The return value (n_batch, n_presyn, n_postsyn) contains the time of on outgoing spike to neuron n_postsyn (for n_batch),
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
    exponentiated_spike_times_syn = torch.exp(input_spikes / tau_syn)
    factor_a1 = torch.cumsum(exponentiated_spike_times_syn * input_weights, dim=1) # n_batch x n_presyn x postsyn

    # to prevent NaNs when first (sorted) weight(s) is 0, thus A and B, and ratio NaN add epsilon
    eps = 1e-6
    factor_b = torch.cumsum(input_spikes * exponentiated_spike_times_syn * input_weights, dim=1) / tau_syn + eps
    factor_c = (thresholds.view(1, 1, n_postsyn) - neuron_params['leak']) * neuron_params['g_leak']
    zForLambertW = -factor_c / factor_a1 * torch.exp(factor_b / factor_a1)

    factor_W = lambertw(zForLambertW, device)

    ret_val = tau_syn * (factor_b / factor_a1 - factor_W)

    return ret_val


def get_spiketime_derivative(input_spikes, input_weights, neuron_params, device,
                             output_spikes, input_delays, thresholds):
    """Calculating the derivatives, see above.

    Weights and (delayed) spike times have shape batch,presyn,postsyn, are ordered according to spike times.
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
    exponentiated_spike_times_syn = torch.exp(input_spikes / tau_syn)

    eps = 1e-6
    factor_a1 = torch.sum(exponentiated_spike_times_syn * causal_weights, dim=1, keepdim=True) # n_batch x n_pre x n_post
    factor_b = torch.sum(input_spikes * exponentiated_spike_times_syn * causal_weights, dim=1, keepdim=True) / tau_syn + eps # dito
    factor_c = (thresholds.view(1, 1, n_postsyn) - neuron_params['leak']) * neuron_params['g_leak']
    zForLambertW = -factor_c / factor_a1 * torch.exp(factor_b / factor_a1) # dito

    factor_W = lambertw(zForLambertW, device) # dito

    dw = -1. / factor_a1 / (1. + factor_W) * exponentiated_spike_times_syn * output_minus_input # n_batch x n_pre x n_post
    # weight gradient not larger since output time t_v is only influenced by weights w_uv
    dt = -1. / factor_a1 / (1. + factor_W) * causal_weights * exponentiated_spike_times_syn * \
        (output_minus_input - tau_syn) / tau_syn # n_batch x n_pre x n_post

    # manually set the uncausal and inf output spike entries 0
    dw[mask] = 0.
    dt[mask] = 0.
    dtheta = torch.zeros_like(dt)
    dd = torch.zeros_like(dt)
    
    return dw, dt, dtheta, dd
