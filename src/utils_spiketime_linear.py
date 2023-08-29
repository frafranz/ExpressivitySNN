#!python3
import torch
import torch.nn
import torch.autograd


torch.set_default_dtype(torch.float64)


def get_spiketime(input_spikes, input_weights, thresholds, neuron_params, device):
    """Calculating spike times, all at once.

    Called from EqualtimeFunction below, for each layer.
    Dimensions are crucial:
        input spikes have dimension batchesxN with N presynaptic neurons
        input weights have dimension BATCHESxNxM with N pre and M postsynaptic neurons.
        thresholds have dimension M, corresponding to all postsynaptic neurons
    The functions relies on the incoming spike time with their respective weights being sorted.

    The return value (n_batch, n_presyn, n_postsyn) contains the time of on outgoing spike to neuron n_postsyn (for n_batch),
    given that all input spikes up to the one at n_presyn have arrived.
    """
    n_batch, n_presyn = input_spikes.shape
    n_batch2, n_presyn2, n_postsyn = input_weights.shape
    assert n_batch == n_batch2, "Deep problem with unequal batch sizes"
    assert n_presyn == n_presyn2

    # fastest implementation: multiply spike times by weights and finally sum over causal spike times
    # set positive lower bound to avoid NaN during division, since a negative (or 0) sum of weights means there is no output pulse
    eps = 1e-10
    summed_weights = torch.clamp(torch.cumsum(input_weights, dim=1), min=eps) # cumulative sum over input spikes
    a = torch.cumsum(input_spikes.unsqueeze(-1)*input_weights, dim=1) # cumulative sum over weighted input spikes

    ret_val = (thresholds.view(1, 1, n_postsyn)+a)/summed_weights # shape: one possible spike time for each batch, last input and output
    ret_val[summed_weights==eps] = torch.inf # if the sum of weights is negative or 0, there won't be an output spike
    
    return ret_val


def get_spiketime_derivative(input_spikes, input_weights, neuron_params, device,
                             output_spikes, thresholds):
    """Calculating the derivatives, see above.

    Input spikes have shape batch,presyn, weights have shape batch,presyn,postsyn, and both are ordered according to input spike times.

    Output spikes
    """
    n_batch, n_presyn = input_spikes.shape
    n_batch2, n_presyn2, n_postsyn = input_weights.shape
    n_batch3, n_postsyn2 = output_spikes.shape
    assert n_batch == n_batch2 == n_batch3, "Deep problem with unequal batch sizes"
    assert n_presyn == n_presyn2
    assert n_postsyn == n_postsyn2

    output_minus_input = -input_spikes.unsqueeze(-1) + output_spikes.view(n_batch, 1, n_postsyn)
    mask = (output_minus_input < 0) | torch.isinf(output_minus_input) | torch.isnan(output_minus_input) # verify causality: output after input, or output never occurs
    # can NaNs occur in the output? yes, for example due to inf - inf
    causal_weights = input_weights
    # set infinities to 0, preventing nans
    causal_weights[mask] = 0.
    input_spikes[torch.isinf(input_spikes)] = 0. # we are not working on the actual spike data but only on a sorted copy from gather
    output_spikes[torch.isinf(output_spikes)] = 0. # TODO: here we use the actual output spike data, problem? (will not use output spike values anymore)

    eps = 1e-10
    summed_weights = torch.clamp(torch.sum(causal_weights, 1, keepdim=True),min=eps) # here we sum only over contributing (causal) weights (negative summed weights would never cause a spike)
    dw = -output_minus_input / summed_weights # for weight gradient, need shape n_batch x n_pre x n_post (since t_v only depends on w_uv)
    dt = causal_weights / summed_weights
    if neuron_params.get("train_threshold"):
        dtheta = 1. / summed_weights.repeat(1,n_presyn,1)
    else:
        dtheta = torch.zeros_like(dt)

    # manually set the uncausal and inf output spike entries 0
    dw[mask] = 0.
    dt[mask] = 0.
    dtheta[output_spikes.unsqueeze(1).repeat(1, n_presyn, 1)<=0] = 0. # for threshold only require that output neuron spikes, independent from input neurons
    
    return dw, dt, dtheta
