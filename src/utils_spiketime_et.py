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
    tmp_mask = torch.tril_indices(n_presyn, n_presyn, offset=-1)  # want diagonal thus offset (once below diagonal)
    weights_split[:, tmp_mask[0], tmp_mask[1], :] = 0. # mask all indices strictly below the diagonal (these won't be needed)

    # temporary reshape for torch reasons (view is like reshape, but ensuring that the data agree and are not copied)
    weights_split = weights_split.view(n_batch, n_presyn, n_presyn * n_postsyn)
    # new (empty) dimension needed for torch reasons
    input_spikes = input_spikes.view(n_batch, 1, n_presyn)

    tau_syn = neuron_params['tau_syn']
    exponentiated_spike_times_syn = torch.exp(input_spikes / tau_syn)

    # to prevent NaNs when first (sorted) weight(s) is 0, thus A and B, and ratio NaN add epsilon
    eps = 1e-6
    factor_a1 = torch.matmul(exponentiated_spike_times_syn, weights_split) # n_batch x 1 x n_pre times n_batch x n_pre x (n_pre n_post)
    # the result has dimensions n_batch x 1 x (n_pre n_post) (same for next row, later returned to n_batch x n_pre x n_post)
    factor_b = torch.matmul(input_spikes * exponentiated_spike_times_syn, weights_split) / tau_syn + eps
    factor_c = (neuron_params['threshold'] - neuron_params['leak']) * neuron_params['g_leak']
    zForLambertW = -factor_c / factor_a1 * torch.exp(factor_b / factor_a1)

    factor_W = lambertw(zForLambertW, device)

    ret_val = tau_syn * (factor_b / factor_a1 - factor_W)
    ret_val = ret_val.view(n_batch, n_presyn, n_postsyn)

    # new implementation:

    # # split up weights for each causal set length (new dimensions: batches x n_pre x causal sets x n_post)
    # weights_split = input_weights[:, :, None, :]
    # weights_split = weights_split.repeat(1, 1, n_presyn, 1)
    # tmp_mask = torch.tril_indices(n_presyn, n_presyn, offset=-1)  # want diagonal thus offset once below diagonal
    # weights_split[:, tmp_mask[0], tmp_mask[1], :] = 0. # mask all indices strictly below the diagonal (these won't be needed)

    # # temporary reshape for torch reasons (view is like reshape, but ensuring that the data agree and are not copied)
    # weights_split = weights_split.view(n_batch, n_presyn, n_presyn * n_postsyn)
    # # new (empty) dimension needed for torch reasons
    # input_spikes = input_spikes.view(n_batch, 1, n_presyn)

    # # added to avoid NaN during division when weights are 0 TODO: does a negative value make sense here? or should they be clamped to > 0?
    # eps = 1e-6
    # summed_weights = torch.sum(weights_split, 1) + eps # here we sum over the cols of the tridiagonal matrix, i.e. first only one, second two etc.
    # a1 = torch.matmul(input_spikes, weights_split) # n_batch x 1 x n_pre times n_batch x n_pre x (n_pre n_post)
    # # the result has dimensions n_batch x 1 x (n_pre n_post) (same for next row, later returned to n_batch x n_pre x n_post)
    
    # ret_val = (neuron_params['threshold']+a1)/summed_weights # TODO: will later adapt to threshold theta_v and add delay d_uv to t_u
    # ret_val = ret_val.view(n_batch, n_presyn, n_postsyn) # ensure correct shape (one possible spike time for each batch and synapse)

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

    output_minus_input = -input_spikes.view(n_batch, n_presyn, 1) + output_spikes.view(n_batch, 1, n_postsyn) # n_batch x n_pre x n_post
    mask = (output_minus_input < 0) | torch.isinf(output_minus_input) | torch.isnan(output_minus_input)
    causal_weights = input_weights
    # set infinities to 0 preventing nans
    causal_weights[mask] = 0.
    input_spikes[torch.isinf(input_spikes)] = 0.
    output_spikes[torch.isinf(output_spikes)] = 0.

    input_spikes = input_spikes.view(n_batch, 1, n_presyn)

    tau_syn = neuron_params['tau_syn']
    exponentiated_spike_times_syn = torch.exp(input_spikes / tau_syn) # n_batch x 1 x n_pre

    eps = 1e-6
    factor_a1 = torch.matmul(exponentiated_spike_times_syn, causal_weights) # n_batch x 1 x n_pre times n_batch x n_pre x n_post = n_batch x 1 x n_post
    factor_b = torch.matmul(input_spikes * exponentiated_spike_times_syn, causal_weights) / tau_syn + eps # dito
    factor_c = (neuron_params['threshold'] - neuron_params['leak']) * neuron_params['g_leak']
    zForLambertW = -factor_c / factor_a1 * torch.exp(factor_b / factor_a1) # dito

    factor_W = lambertw(zForLambertW, device) # dito

    exponentiated_spike_times_syn = exponentiated_spike_times_syn.squeeze().unsqueeze(-1) # n_batch x n_pre x 1

    dw = -1. / factor_a1 / (1. + factor_W) * exponentiated_spike_times_syn * output_minus_input # n_batch x 1 x n_post times n_batch x n_pre x 1 = n_batch x n_pre x n_post
    # weight gradient not larger since output time t_v is only influenced by weights w_uv
    dt = -1. / factor_a1 / (1. + factor_W) * causal_weights * exponentiated_spike_times_syn * \
        (output_minus_input - tau_syn) / tau_syn # n_batch x 1 x n_post times n_batch x n_pre x n_post = n_batch x n_pre x n_post

    # manually set the uncausal and inf output spike entries 0
    dw[mask] = 0.
    dt[mask] = 0.

    # new implementation:

    # output_minus_input = -input_spikes.view(n_batch, n_presyn, 1) + output_spikes.view(n_batch, 1, n_postsyn)
    # mask = (output_minus_input < 0) | torch.isinf(output_minus_input) | torch.isnan(output_minus_input) # verify causality: output after input, or output never occurs
    # # can NaNs occur in the output? yes, for example inf - inf
    # causal_weights = input_weights
    # # set infinities to 0 preventing nans
    # causal_weights[mask] = 0.
    # input_spikes[torch.isinf(input_spikes)] = 0. # we are not working on the actual spike data but only on a sorted copy from gather
    # output_spikes[torch.isinf(output_spikes)] = 0. # TODO: here we use the actual output spike data, problem? (will not use output spike values anymore)

    # input_spikes = input_spikes.view(n_batch, 1, n_presyn)

    # exponentiated_spike_times_syn = torch.exp(input_spikes / tau_syn)

    # eps = 1e-6
    # summed_weights = torch.sum(causal_weights, 1, keepdim=True) + eps # here we sum only over contributing (causal) weights
    # dw = output_minus_input / summed_weights # for weight gradient, need shape n_batch x n_pre x n_post (since t_v only depends on w_uv) # TODO: would add in delay here
    # dt = causal_weights / summed_weights
    # # dd = causal_weights / summed_weights # also provide derivative wrt delay
    # # dtheta = 1./summed_weights

    # # manually set the uncausal and inf output spike entries 0
    # dw[mask] = 0.
    # dt[mask] = 0.
    # # dtheta[mask] = 0.
    # # dd[mask] = 0.
    
    return dw, dt #dtheta,dd
