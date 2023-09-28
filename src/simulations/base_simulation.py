import numpy as np
import torch
import sys

from utils import to_device

torch.set_default_dtype(torch.float64)

class BaseSimulation(torch.autograd.Function):
    """Class that provides input verification and gradient computation applicable both in explicit and time-discretized simulation"""
    
    @staticmethod
    def prepare_inputs(input_spikes, input_weights, input_delays, thresholds, sim_params):
        """Verify input dimensions and process possible delay values.

        Arguments:
            input_spikes, input_weights, input_delays, thresholds: input that is used for calculations
            sim_params: parameters of the simulation and training

        Returns:
            n_batch:                number of batches
            n_postsyn:              number of output neurons
            delayed_input_spikes:   inputs with possibly added delays
        """
        n_batch, n_presyn = input_spikes.shape
        n_presyn2, n_postsyn = input_weights.shape
        n_presyn3, n_postsyn2 = input_delays.shape
        n_postsyn3 = thresholds.shape[0]
        assert n_presyn == n_presyn2 == n_presyn3
        assert n_postsyn == n_postsyn2 == n_postsyn3


        if sim_params['activation'] in ['alpha_equaltime', 'alpha_doubletime'] and (sim_params.get('train_delay') or sim_params.get('train_threshold')):
            raise NotImplementedError(f"training of delay and threshold not implemented for {sim_params['activation']}")

        if sim_params.get('substitute_delay'): # substitute delays by their logarithms to enforce positivity
            delayed_input_spikes = input_spikes.unsqueeze(-1) + torch.exp(input_delays.unsqueeze(0))
        else:
            delayed_input_spikes = input_spikes.unsqueeze(-1) + input_delays.unsqueeze(0)

        return n_batch, n_postsyn, delayed_input_spikes
    

    @staticmethod
    def backward(ctx, propagated_error, spike_contribution_error):
        """
        Custom backward propagation function.

        It must accept a context ctx as the first argument, followed by as many outputs as the forward() returned.
        (None will be passed in for non tensor outputs of the forward function)
        It should return as many tensors as there were inputs to forward(). Each argument is the gradient w.r.t the given output, 
        and each returned value should be the gradient w.r.t. the corresponding input. 
        If an input is not a Tensor or is a Tensor not requiring grads, you can just pass None as a gradient for that input.
        (cf. TORCH.AUTOGRAD.FUNCTION.BACKWARD)

        Arguments:
            ctx: context variable from torch
            propapagated error: gradients with respect to output spike times to next layer
            spike_contribution_error: error with respect to spike contributions, which is not used in the backward pass

        Returns:
            new_propagated_error: derivatives w.r.t. input spikes times
            weight_gradient: derivatives w.r.t. weights
            delay_gradient: derivatives w.r.t. delays
            None, None: no derivatives needed w.r.t. neuron_params and device
        """
        # recover saved values
        input_spikes, input_weights, input_delays, thresholds, output_spikes = ctx.saved_tensors
        
        # might be left out, since already done before saving tensors, but like this we also know the indices (need to revert later)
        if ctx.sim_params.get('substitute_delay'): # substitute delays by their logarithms to enforce positivity
            delayed_input_spikes = input_spikes.unsqueeze(-1) + torch.exp(input_delays.unsqueeze(0))
        else:
            delayed_input_spikes = input_spikes.unsqueeze(-1) + input_delays.unsqueeze(0)

        sort_indices = delayed_input_spikes.argsort(1)
        sorted_spikes = delayed_input_spikes.gather(1, sort_indices)
        n_batch = input_spikes.shape[0]
        sorted_weights = torch.gather(input_weights.unsqueeze(0).expand(n_batch,-1,-1), dim=1, index=sort_indices) # output is of shape n_batch x n_pre x n_post
        # with missing label spikes the propagated error can be nan
        propagated_error[torch.isnan(propagated_error)] = 0 # shape of propagated error is n_batch x n_post

        batch_size = input_spikes.shape[0]
        number_inputs = input_weights.shape[0]
        
        if ctx.sim_params['activation']=='linear':
            import spiketimes.piecewise_linear as spiketime # use our linear activation model
        elif ctx.sim_params['activation'] == 'alpha_equaltime':
            import spiketimes.alpha_1 as spiketime # use the alpha activation as in Goeltz et al
        elif ctx.sim_params['activation'] == 'alpha_doubletime':
            import spiketimes.alpha_2 as spiketime # use the modified alpha activation as in Goeltz et al
        else:
            raise NotImplementedError(f"optimizer {ctx.sim_params['activation']} not implemented")

        dw_ordered, dt_ordered, dd_ordered, dtheta = spiketime.get_spiketime_derivative(
                sorted_spikes, sorted_weights, ctx.sim_params, ctx.device, output_spikes, input_delays, thresholds)

        # retransform it in the correct way
        dw = to_device(torch.zeros(dw_ordered.size()), ctx.device)
        dt = to_device(torch.zeros(dt_ordered.size()), ctx.device)
        dd = to_device(torch.zeros(dd_ordered.size()), ctx.device)
        dtheta = to_device(dtheta, ctx.device)
        # masking: determine how to revert sorting of spike times (this depends on the output neuron since the delays are different!)
        mask_from_spikeordering = torch.argsort(sort_indices, dim=1)

        dw = torch.gather(dw_ordered, 1, mask_from_spikeordering)
        dt = torch.gather(dt_ordered, 1, mask_from_spikeordering)
        dd = torch.gather(dd_ordered, 1, mask_from_spikeordering)
        # no ordering needed for dtheta since it does not refer to input spike order

        error_to_work_with = propagated_error.view(
            propagated_error.shape[0], 1, propagated_error.shape[1]) # reshape propagated error to n_batch x 1 x n_post

        weight_gradient = dw * error_to_work_with # chain rule: weight gradient is derivative of outgoing spike times wrt weights times derivative wrt outoing spikes
        delay_gradient = dd * error_to_work_with
        threshold_gradient = dtheta * error_to_work_with

        if ctx.sim_params['max_dw_norm'] is not None: # TODO: could adapt the weight clipping
            """ to prevent large weight changes, we identify output spikes with a very large dw
            this happens when neuron barely spikes, aka small changes determine whether it spikes or not
            technically, the membrane maximum comes close to the threshold,
            and the derivative at the threshold will vanish.
            as the derivative here are wrt the times, kinda a switch of axes (see LambertW), # TODO: do not use LambertW so think about this in our case
            the derivatives will diverge in those cases."""
            weight_gradient_norms, _ = weight_gradient.abs().max(dim=1) # calculate max gradient for every batch and output neuron (i.e. over input neurons)
            weight_gradient_jumps = weight_gradient_norms > ctx.sim_params['max_dw_norm'] # get batch / output neuron pairs for which there is a high gradient
            if weight_gradient_jumps.sum() > 0:
                print(f"gradients too large (input size {number_inputs}), chopped the following:"
                      f"{weight_gradient_norms[weight_gradient_jumps]}")
            weight_gradient = weight_gradient.permute([0, 2, 1])
            weight_gradient[weight_gradient_jumps] = 0. # permuting allows us to access the batch and output neuron with gradient_jumps
            weight_gradient = weight_gradient.permute([0, 2, 1]) # permute back to previous

            # # TODO: could copy the same steps for delay_gradient
            # delay_gradient_norms, _ = delay_gradient.abs().max(dim=1)
            # delay_gradient_jumps = delay_gradient_norms > ctx.sim_params['max_dw_norm'] # TODO: if this clipping should be used for delay gradients, save max_dd_norm in config
            # if delay_gradient_jumps.sum() > 0:
            #     print(f"gradients too large (input size {number_inputs}), chopped the following:"
            #           f"{delay_gradient_norms[delay_gradient_jumps]}")
            # delay_gradient = delay_gradient.permute([0, 2, 1])
            # delay_gradient[delay_gradient_jumps] = 0.
            # delay_gradient = delay_gradient.permute([0, 2, 1])

        # averaging over batches to get final update
        weight_gradient = weight_gradient.sum(0) # now only n_pre x n_post
        delay_gradient = delay_gradient.sum(0) # now only n_pre x n_post; the outgoing gradients are summed over the batch, the ones handed on aren't
        threshold_gradient = threshold_gradient.sum(0)

        new_propagated_error = torch.bmm(
            dt, # n_batch x n_pre x n_post
            error_to_work_with.permute(0, 2, 1) # n_batch x n_post x 1 (result: n_batch x n_pre x 1)
        ).view(batch_size, number_inputs) # chain rule: gradient wrt incoming spike times are gradients wrt outoing ones times the gradients within this layer

        if torch.any(torch.isinf(weight_gradient)) or \
           torch.any(torch.isinf(new_propagated_error)) or \
           torch.any(torch.isnan(weight_gradient)) or \
           torch.any(torch.isnan(new_propagated_error)) or \
           torch.any(torch.isinf(delay_gradient)) or \
           torch.any(torch.isinf(threshold_gradient)) or \
           torch.any(torch.isnan(delay_gradient)) or \
           torch.any(torch.isnan(threshold_gradient)):
            print(f" wg nan {torch.isnan(weight_gradient).sum()}, inf {torch.isinf(weight_gradient).sum()}")
            print(f" new_propagated_error nan {torch.isnan(new_propagated_error).sum()}, "
                  f"inf {torch.isinf(new_propagated_error).sum()}")
            print(f" dg nan {torch.isnan(delay_gradient).sum()}, inf {torch.isinf(delay_gradient).sum()}")
            print(f" tg nan {torch.isnan(threshold_gradient).sum()}, inf {torch.isinf(threshold_gradient).sum()}")
            print('found nan or inf in propagated_error, weight_gradient, delay_gradient or threshold_gradient, something is wrong oO')
            sys.exit()
        return new_propagated_error, weight_gradient, delay_gradient, threshold_gradient, None, None