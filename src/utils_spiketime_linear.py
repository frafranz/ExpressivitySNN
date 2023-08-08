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
    The functions relies on the incoming spike time with their respective weights being sorted.

    The return value (n_batch, n_presyn, n_postsyn) contains the time of on outgoing spike to neuron n_postsyn (for n_batch),
    given that all input spikes up to the one at n_presyn have arrived.
    """
    n_batch, n_presyn = input_spikes.shape
    n_batch2, n_presyn2, n_postsyn = input_weights.shape
    assert n_batch == n_batch2, "Deep problem with unequal batch sizes"
    assert n_presyn == n_presyn2

    # split up weights for each causal set length (new dimensions: batches x n_pre x causal sets x n_post)
    weights_split = input_weights[:, :, None, :]
    weights_split = weights_split.repeat(1, 1, n_presyn, 1)
    tmp_mask = torch.tril_indices(n_presyn, n_presyn, offset=-1)  # want diagonal thus offset once below diagonal
    weights_split[:, tmp_mask[0], tmp_mask[1], :] = 0. # mask all indices strictly below the diagonal (these won't be needed)

    # temporary reshape for torch reasons (view is like reshape, but ensuring that the data agree and are not copied)
    weights_split = weights_split.view(n_batch, n_presyn, n_presyn * n_postsyn)
    # new (empty) dimension needed for torch reasons
    input_spikes = input_spikes.view(n_batch, 1, n_presyn)

    # added to avoid NaN during division when weights are 0 TODO: does a negative value make sense here? or should they be clamped to > 0?
    eps = 1e-10
    summed_weights = torch.sum(weights_split, 1) + eps # here we sum over the cols of the tridiagonal matrix, i.e. first only one, second two etc.
    a1 = torch.matmul(input_spikes, weights_split).squeeze(1) # n_batch x 1 x n_pre times n_batch x n_pre x (n_pre n_post)
    # the result has dimensions n_batch x 1 x (n_pre n_post) (same for next row, later casted to n_batch x n_pre x n_post)
    
    ret_val = (neuron_params['threshold']+a1)/summed_weights # TODO: will later adapt to threshold theta_v and add delay d_uv to t_u
    ret_val = ret_val.view(n_batch, n_presyn, n_postsyn) # ensure correct shape (one possible spike time for each batch and synapse)

    return ret_val


def get_spiketime_derivative(input_spikes, input_weights, neuron_params, device,
                             output_spikes):
    """Calculating the derivatives, see above.

    Weights have shape batch,presyn,postsyn, are ordered according to spike times
    """
    n_batch, n_presyn = input_spikes.shape
    n_batch2, n_presyn2, n_postsyn = input_weights.shape
    assert n_batch == n_batch2, "Deep problem with unequal batch sizes"
    assert n_presyn == n_presyn2

    output_minus_input = -input_spikes.view(n_batch, n_presyn, 1) + output_spikes.view(n_batch, 1, n_postsyn)
    mask = (output_minus_input < 0) | torch.isinf(output_minus_input) | torch.isnan(output_minus_input) # verify causality: output after input, or output never occurs
    # can NaNs occur in the output? yes, for example inf - inf
    causal_weights = input_weights
    # set infinities to 0 preventing nans
    causal_weights[mask] = 0.
    input_spikes[torch.isinf(input_spikes)] = 0. # we are not working on the actual spike data but only on a sorted copy from gather
    output_spikes[torch.isinf(output_spikes)] = 0. # TODO: here we use the actual output spike data, problem? (will not use output spike values anymore)

    input_spikes = input_spikes.view(n_batch, 1, n_presyn)

    eps = 1e-10
    summed_weights = torch.sum(causal_weights, 1, keepdim=True) + eps # here we sum only over contributing (causal) weights
    dw = -output_minus_input / summed_weights # for weight gradient, need shape n_batch x n_pre x n_post (since t_v only depends on w_uv) # TODO: would add in delay here
    dt = causal_weights / summed_weights
    # dd = causal_weights / summed_weights # also provide derivative wrt delay
    # dtheta = 1./summed_weights

    # manually set the uncausal and inf output spike entries 0
    dw[mask] = 0.
    dt[mask] = 0.
    # dtheta[mask] = 0.
    # dd[mask] = 0.
    
    return dw, dt #dtheta,dd
