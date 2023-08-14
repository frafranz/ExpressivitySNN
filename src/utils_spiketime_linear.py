#!python3
import torch
import torch.nn
import torch.autograd


torch.set_default_dtype(torch.float64)


def get_spiketime(input_spikes, input_weights, neuron_params, device):
    """Calculating spike times, all at once.

    Called from EqualtimeFunction below, for each layer.
    Dimensions are crucial:
        input weights have dimension BATCHESxNxM with N pre and M postsynaptic neurons.
        (delayed) input spikes also have dimension batchesxNxM, where the last coordinate represents the output neuron for which the input is considered
    The functions relies on the incoming spike time with their respective weights being sorted, delays between neurons are already accounted for.

    The return value (n_batch, n_presyn, n_postsyn) contains the time of on outgoing spike to neuron n_postsyn (for n_batch),
    given that all input spikes up to the one at n_presyn have arrived.
    """
    n_batch, n_presyn, n_postsyn = input_spikes.shape
    n_batch2, n_presyn2, n_postsyn2 = input_weights.shape
    assert n_batch == n_batch2, "Deep problem with unequal batch sizes"
    assert n_presyn == n_presyn2
    assert n_postsyn == n_postsyn2

    # split up weights for each causal set length (new dimensions: batches x n_pre x causal sets x n_post)
    weights_split = input_weights[:, :, None, :]
    weights_split = weights_split.repeat(1, 1, n_presyn, 1)
    tmp_mask = torch.tril_indices(n_presyn, n_presyn, offset=-1)  # want diagonal thus offset once below diagonal
    weights_split[:, tmp_mask[0], tmp_mask[1], :] = 0. # mask all indices strictly below the diagonal (these won't be needed)

    # temporary reshape for torch reasons (view is like reshape, but ensuring that the data agree and are not copied)
    weights_split = weights_split.view(n_batch, n_presyn, n_presyn * n_postsyn)

    # fastest implementation: multiply delayed spike times by weights and finally sum over causal spike times
    weighted_delayed_spikes_split = (input_spikes*input_weights)[:, :, None, :]
    weighted_delayed_spikes_split = weighted_delayed_spikes_split.repeat(1, 1, n_presyn, 1)
    tmp_mask = torch.tril_indices(n_presyn, n_presyn, offset=-1)  # want diagonal thus offset once below diagonal
    weighted_delayed_spikes_split[:, tmp_mask[0], tmp_mask[1], :] = 0. # mask all indices strictly below the diagonal (these won't be needed)
    # temporary reshape for torch reasons (view is like reshape, but ensuring that the data agree and are not copied)
    weighted_delayed_spikes_split = weighted_delayed_spikes_split.view(n_batch, n_presyn, n_presyn * n_postsyn)
    
    a = torch.sum(weighted_delayed_spikes_split, 1) # n_batch x (n_pre n_post)
    # set positive lower bound to avoid NaN during division, since a negative (or 0) sum of weights means there is no output pulse
    eps = 1e-10
    summed_weights = torch.clamp(torch.sum(weights_split, 1), min=eps) # here we sum over the cols of the tridiagonal matrix, i.e. first only one, second two etc.
    # has shape n_batch x (n_pre x n_post)

    ret_val = (neuron_params['threshold']+a)/summed_weights # TODO: will later adapt to threshold theta_v
    ret_val[summed_weights==eps] = torch.inf # if the sum of weights is negative or 0, there won't be an output spike
    ret_val = ret_val.view(n_batch, n_presyn, n_postsyn) # ensure correct shape (one possible spike time for each batch and synapse)

    return ret_val


def get_spiketime_derivative(input_spikes, input_weights, neuron_params, device,
                             output_spikes):
    """Calculating the derivatives, see above.

    Weights and (delayed) input spikes have shape batch,presyn,postsyn, are ordered according to spike times
    """
    n_batch, n_presyn, n_postsyn = input_spikes.shape
    n_batch2, n_presyn2, n_postsyn2 = input_weights.shape
    n_batch3, n_postsyn3 = output_spikes.shape
    assert n_batch == n_batch2 == n_batch3, "Deep problem with unequal batch sizes"
    assert n_presyn == n_presyn2
    assert n_postsyn == n_postsyn2 == n_postsyn3

    output_minus_input = -input_spikes + output_spikes.view(n_batch, 1, n_postsyn)
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
    if "train_delay" in neuron_params and neuron_params["train_delay"]:
        dd = causal_weights / summed_weights
    else:
        dd = torch.zeros_like(dt)
    # dtheta = 1./summed_weights # TODO: also add threshold gradient

    # manually set the uncausal and inf output spike entries 0
    dw[mask] = 0.
    dt[mask] = 0.
    dd[mask] = 0.
    # dtheta[mask] = 0.
    
    return dw, dt, dd #dtheta
