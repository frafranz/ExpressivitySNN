#!python3
import numpy as np
import os.path as osp
import sys
import time
import torch
import torch.nn
import torch.autograd

torch.set_default_dtype(torch.float64)


class EqualtimeFunctionEventbased(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                input_spikes, input_weights, input_delays, thresholds,
                neuron_params, device):
        """Class that calculates the EventBased spikes, and provides backward function

        Arguments:
            ctx: from torch, used for saving for backward pass
            input_spikes, input_weights, input_delays, thresholds: input that is used for calculations
            neuron_params: constants used in calculation
            device: torch specifics, for GPU use

        Returns:
            output_spikes:  outgoing spike times
        """
        n_batch, n_presyn = input_spikes.shape
        n_presyn2, n_postsyn = input_weights.shape
        n_presyn3, n_postsyn2 = input_delays.shape
        n_postsyn3 = thresholds.shape[0]
        assert n_presyn == n_presyn2 == n_presyn3
        assert n_postsyn == n_postsyn2 == n_postsyn3

        # create causal set
        # due to to different delays, the sorting has to be done for every output separately
        if neuron_params.get('substitute_delay'): # substitute delays by their logarithms to enforce positivity
            delayed_input_spikes = input_spikes.unsqueeze(-1) + torch.exp(input_delays.unsqueeze(0))
        else:
            delayed_input_spikes = input_spikes.unsqueeze(-1) + input_delays.unsqueeze(0)
        sort_indices = delayed_input_spikes.argsort(1)
        sorted_spikes = delayed_input_spikes.gather(1, sort_indices)
        sorted_weights = torch.gather(input_weights.unsqueeze(0).expand(n_batch,-1,-1), dim=1, index=sort_indices) # output is of shape n_batch x n_pre x n_post
        # (in each batch) set weights=0 and spiketime=0 for neurons with inf spike time to prevent nans
        mask_of_inf_spikes = torch.isinf(sorted_spikes)
        sorted_spikes_masked = sorted_spikes.clone().detach() # TODO: why detach here? means that these spikes are not part of the comp. graph, i.e. not trained?
        sorted_spikes_masked[mask_of_inf_spikes] = 0.
        sorted_weights[mask_of_inf_spikes] = 0. # TODO: why no clone, detach for weights? (means setting them to 0 is actually part of the comp. graph?)
        
        output_spikes = to_device(torch.ones(input_weights.size()) * np.inf, device) # prepares the array for the output spikes on the device

        if neuron_params['activation']=='linear':
            import utils_spiketime_linear as utils_spiketime # use our linear activation model
        elif neuron_params['activation']=='alpha_equaltime':
            if neuron_params['train_delay'] or neuron_params['train_threshold']:
                raise NotImplementedError(f"training of delay and threshold not implemented for {neuron_params['activation']}")
            import utils_spiketime_et as utils_spiketime # use the alpha activation as in Goeltz et al
        elif neuron_params['activation']=='alpha_doubletime':
            if neuron_params['train_delay'] or neuron_params['train_threshold']:
                raise NotImplementedError(f"training of delay and threshold not implemented for {neuron_params['activation']}")
            import utils_spiketime_dt as utils_spiketime # use the modified alpha activation as in Goeltz et al
        else:
            raise NotImplementedError(f"optimizer {neuron_params['activation']} not implemented")

        tmp_output = utils_spiketime.get_spiketime(
            sorted_spikes_masked,
            sorted_weights,
            thresholds,
            neuron_params, device)

        # compare new spike times with previous, to set all cases in which no spike would occur to inf
        before_last_input = tmp_output < sorted_spikes # output cannot happen before the last required input (means we need less inputs)
        after_next_input = tmp_output > sorted_spikes.roll(-1, dims=1) # compare to next input spike, output must happen before (else we need more inputs)
        after_next_input[:, -1, :] = 0.  # last has no subsequent spike, so the output spike is always early enough

        # set non-causal times to inf (so that they are ignored in the min taken to find the output times)
        tmp_output[before_last_input] = float('inf')
        tmp_output[after_next_input] = float('inf')

        output_spikes, causal_set_lengths = torch.min(tmp_output, dim=1) # take output spike happening after a minimum of incoming spikes (complexity: n_pre)

        ctx.sim_params = neuron_params
        ctx.device = device
        ctx.save_for_backward(
            input_spikes, input_weights, input_delays, thresholds, output_spikes)
        if torch.isnan(output_spikes).sum() > 0:
            raise ArithmeticError("There are NaNs in the output times, this means a serious error occured")
        return output_spikes

    @staticmethod
    def backward(ctx, propagated_error):
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
            import utils_spiketime_linear as utils_spiketime # use our linear activation model
        elif ctx.sim_params['activation']=='alpha_equaltime':
            import utils_spiketime_et as utils_spiketime # use the alpha activation as in Goeltz et al
        elif ctx.sim_params['activation']=='alpha_doubletime':
            import utils_spiketime_dt as utils_spiketime # use the modified alpha activation as in Goeltz et al
        else:
            raise NotImplementedError(f"optimizer {ctx.sim_params['activation']} not implemented")

        dw_ordered, dt_ordered, dd_ordered, dtheta = utils_spiketime.get_spiketime_derivative(
            sorted_spikes, sorted_weights, ctx.sim_params, ctx.device, output_spikes, input_delays)

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

        if ctx.sim_params['max_dw_norm'] is not None: # TODO: ignored for now, adapt once we are training the new model
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

            # # TODO: for now just blindly copy the same steps for delay_gradient
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
           torch.any(torch.isinf(delay_gradient)) or \
           torch.any(torch.isinf(threshold_gradient)) or \
           torch.any(torch.isnan(weight_gradient)) or \
           torch.any(torch.isnan(new_propagated_error)) or \
           torch.any(torch.isnan(weight_gradient)) or \
           torch.any(torch.isnan(threshold_gradient)):
            print(f" wg nan {torch.isnan(weight_gradient).sum()}, inf {torch.isinf(weight_gradient).sum()}")
            print(f" dg nan {torch.isnan(delay_gradient).sum()}, inf {torch.isinf(delay_gradient).sum()}")
            print(f" tg nan {torch.isnan(threshold_gradient).sum()}, inf {torch.isinf(threshold_gradient).sum()}")
            print(f" new_propagated_error nan {torch.isnan(new_propagated_error).sum()}, "
                  f"inf {torch.isinf(new_propagated_error).sum()}")
            print('found nan or inf in propagated_error, weight_gradient, delay_gradient or threshold_gradient, something is wrong oO')
            sys.exit()

        return new_propagated_error, weight_gradient, delay_gradient, threshold_gradient, None, None


class EqualtimeFunctionIntegrator(EqualtimeFunctionEventbased):
    @staticmethod
    def forward(ctx,
                input_spikes, input_weights, input_delays, thresholds,
                sim_params, device, output_times=None):
        """use a simple euler integration, then compare with a threshold to determine spikes"""
        batch_size, input_features = input_spikes.shape
        _, output_features = input_weights.shape

        # TODO: ignoring delays and thresholds so far
        input_transf = torch.zeros(tuple(input_spikes.shape) + (sim_params['steps'], ),
                                   device=device, requires_grad=False)
        input_times_step = (input_spikes / sim_params['resolution']).long()
        input_times_step[input_times_step > sim_params['steps'] - 1] = sim_params['steps'] - 1
        input_times_step[torch.isinf(input_spikes)] = sim_params['steps'] - 1

        # one-hot code input times for easier multiplication
        input_transf = torch.eye(sim_params['steps'], device=device)[
            input_times_step].reshape((batch_size, input_features, sim_params['steps']))

        charge = torch.einsum("abc,bd->adc", (input_transf, input_weights))

        # init is no synaptic current and mem at leak
        syn = torch.zeros((batch_size, output_features), device=device)
        mem = torch.ones((batch_size, output_features), device=device) * sim_params['leak']

        plotting = False
        all_mem = []
        # want to save spikes
        output_spikes = torch.ones((batch_size, output_features), device=device) * float('inf')
        for step in range(sim_params['steps']):
            # print(step)
            mem = sim_params['decay_mem'] * (mem - sim_params['leak']) \
                + 1. / sim_params['g_leak'] * syn * sim_params['resolution'] + sim_params['leak']
            syn = sim_params['decay_syn'] * syn + charge[:, :, step]

            # mask is a logical_and implemented by multiplication
            output_spikes[
                (torch.isinf(output_spikes) *
                 mem > sim_params['threshold'])] = step * sim_params['resolution']
            # reset voltage after spike for plotting
            # mem[torch.logical_not(torch.isinf(output_spikes))] = 0.
            if plotting:
                all_mem.append(mem.numpy())

        if plotting:
            import matplotlib.pyplot as plt
            import warnings
            if batch_size >= 9:
                fig, axes = plt.subplots(3, 3, figsize=(16, 10))
            else:
                fig, ax = plt.subplots(1, 1)
                axes = np.array([ax])
            all_mem = np.array(all_mem)
            np.save("membrane_trace.npy", all_mem)
            np.save("membrane_spike.npy", output_spikes)
            batch_to_plot = 0
            for batch_to_plot, ax in enumerate(axes.flatten()):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    ax.plot(np.arange(sim_params['steps']) * sim_params['resolution'],
                            all_mem[:, batch_to_plot, :])

                    ax.axhline(sim_params['leak'], color='black', lw=0.4)
                    ax.axhline(sim_params['threshold'], color='black', lw=1)
                    for i, sp in enumerate(output_spikes[batch_to_plot]):
                        ax.axvline(sp, color=f"C{i}", ls="-.", ymax=0.5)
                    if output_times is not None:
                        for i, ti in enumerate(output_times[batch_to_plot]):
                            ax.axvline(ti, color=f"C{i}", ls=":", ymin=0.5)

                    ax.set_ylim(
                        (sim_params['threshold'] - sim_params['leak']) * np.array((-1, 1.1)) + sim_params['leak'])

                    ax.set_ylabel(f"C{batch_to_plot}", fontweight='bold')
                    ax.yaxis.label.set_color(f"C{batch_to_plot}")
            fig.tight_layout()
            fig.savefig('debug_int.png')
            plt.close(fig)

        if torch.isnan(output_spikes).sum() > 0:
            raise ArithmeticError("There are NaNs in the output times, this means a serious error occured")

        ctx.sim_params = sim_params
        ctx.device = device
        ctx.save_for_backward(
            input_spikes, input_weights, output_spikes)

        return output_spikes


class EqualtimeLayer(torch.nn.Module):
    def __init__(self, input_features, output_features, sim_params, weights_init, delays_init, thresholds_init,
                 device, bias=0):
        """Setup up a layer of neurons

        Arguments:
            input_features, output_features: number of inputs/outputs
            sim_params: parameters used for simulation
            weights_init: if tuple it is understood as two lists of mean and std, otherwise an array of weights
            delays_init: if tuple it is understood as two lists of mean and std, otherwise an array of weights
            thresholds_init: if tuple it is understood as two lists of mean and std, otherwise an array of weights
            device: torch, gpu stuff
            bias: number of bias inputs
        """
        super(EqualtimeLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.sim_params = sim_params
        self.bias = bias
        self.device = device
        self.use_forward_integrator = sim_params.get('use_forward_integrator', False)
        if self.use_forward_integrator:
            assert 'resolution' in sim_params and 'sim_time' in sim_params
            self.sim_params['steps'] = int(np.ceil(sim_params['sim_time'] / sim_params['resolution']))
            self.sim_params['decay_syn'] = float(np.exp(-sim_params['resolution'] / sim_params['tau_syn']))
            self.sim_params['decay_mem'] = float(np.exp(-sim_params['resolution'] / sim_params['tau_syn']))

        self.weights = torch.nn.Parameter(torch.Tensor(input_features + bias, output_features))
        self.delays = torch.nn.Parameter(torch.Tensor(input_features + bias, output_features))
        self.thresholds = torch.nn.Parameter(torch.Tensor(output_features))

        if isinstance(weights_init, tuple):
            self.weights.data.normal_(weights_init[0], weights_init[1])
        else:
            assert weights_init.shape == (input_features + bias, output_features)
            self.weights.data = weights_init
        
        if isinstance(delays_init, tuple):
            if self.sim_params.get('substitute_delay'):
                self.delays.data.normal_(delays_init[0], delays_init[1])
            else:
                self.delays.data.normal_(delays_init[0], delays_init[1])
        else:
            assert delays_init.shape == (input_features + bias, output_features)
            self.delays.data = delays_init

        if isinstance(thresholds_init, tuple):
            self.thresholds.data.normal_(thresholds_init[0], thresholds_init[1])
        else:
            assert thresholds_init.shape[0] == output_features
            self.thresholds.data = thresholds_init
        

    def forward(self, input_times, output_times=None):
        # depending on configuration use either eventbased, integrator or the hardware
        assert output_times is None
        if self.use_forward_integrator:
            return EqualtimeFunctionIntegrator.apply(input_times, self.weights, self.delays, self.thresholds,
                                                     self.sim_params,
                                                     self.device)
        else:
            return EqualtimeFunctionEventbased.apply(input_times, self.weights, self.delays, self.thresholds,
                                                     self.sim_params,
                                                     self.device)


def bias_inputs(number_biases, t_bias=[0.05]):
    assert len(t_bias) == number_biases
    times = [t_bias[i] for i in range(number_biases)]
    return torch.tensor(times)


def network_load(path, basename, device):
    net = to_device(torch.load(osp.join(path, basename + "_network.pt"),
                               map_location=device),
                    device)
    net.device = get_default_device()
    return net


class LossFunction(torch.nn.Module):
    def __init__(self, number_labels, tau_syn, xi, alpha, beta, device):
        super().__init__()
        self.number_labels = number_labels
        self.tau_syn = tau_syn
        self.xi = xi
        self.alpha = alpha
        self.beta = beta
        self.device = device
        return

    def forward(self, label_times, true_label):
        label_idx = to_device(true_label.clone().type(torch.long).view(-1, 1), self.device)
        true_label_times = label_times.gather(1, label_idx).flatten()
        # cross entropy between true label distribution and softmax-scaled label spike times
        loss = torch.log(torch.exp(-1 * label_times / (self.xi * self.tau_syn)).sum(1)) \
            + true_label_times / (self.xi * self.tau_syn)
        regulariser = self.alpha * torch.exp(true_label_times / (self.beta * self.tau_syn))
        total = loss + regulariser
        total[true_label_times == np.inf] = 100.
        return total.mean()

    def select_classes(self, outputs):
        firsts = outputs.argmin(1)
        firsts_reshaped = firsts.view(-1, 1)
        # count how many firsts had inf or nan as value
        nan_mask = torch.isnan(torch.gather(outputs, 1, firsts_reshaped)).flatten()
        inf_mask = torch.isinf(torch.gather(outputs, 1, firsts_reshaped)).flatten()
        # set firsts to -1 so that they cannot be counted as correct
        firsts[nan_mask] = -1
        firsts[inf_mask] = -1
        return firsts


class LossFunctionMSE(torch.nn.Module):
    def __init__(self, number_labels, tau_syn, correct, wrong, device):
        super().__init__()
        self.number_labels = number_labels
        self.tau_syn = tau_syn
        self.device = device

        self.t_correct = self.tau_syn * correct
        self.t_wrong = self.tau_syn * wrong
        return

    def forward(self, label_times, true_label):
        # get a vector which is 1 at the true label and set to t_correct at the true label and to t_wrong at the others
        target = to_device(torch.eye(self.number_labels), self.device)[true_label.int()] * (self.t_correct - self.t_wrong) + self.t_wrong
        loss = 1. / 2. * (label_times - target)**2
        loss[label_times == np.inf] = 100.
        return loss.mean()

    def select_classes(self, outputs):
        closest_to_target = torch.abs(outputs - self.t_correct).argmin(1)
        ctt_reshaped = closest_to_target.view(-1, 1)
        # count how many firsts had inf or nan as value
        nan_mask = torch.isnan(torch.gather(outputs, 1, ctt_reshaped)).flatten()
        inf_mask = torch.isinf(torch.gather(outputs, 1, ctt_reshaped)).flatten()
        # set firsts to -1 so that they cannot be counted as correct
        closest_to_target[nan_mask] = -1
        closest_to_target[inf_mask] = -1
        return closest_to_target


def GetLoss(training_params, number_labels, tau_syn, device):
    """Dynamically get the loss function depending on the params"""
    # to be downward compatible
    if 'loss' in training_params:
        params = training_params['loss']
    else:
        params = {
            'type': 'TTFS',
            'alpha': training_params['alpha'],
            'beta': training_params['beta'],
            'xi': training_params['xi'],
        }
    if params['type'] == 'TTFS':
        return LossFunction(number_labels, tau_syn,
                            params['xi'], params['alpha'], params['beta'],
                            device)
    elif params['type'] == 'MSE':
        return LossFunctionMSE(number_labels, tau_syn,
                               params['t_correct'], params['t_wrong'],
                               device)
    else:
        raise NotImplementedError(f"loss of type '{params['type']}' not implemented")


def get_default_device():
    # Pick GPU if avialable, else CPU
    if torch.cuda.is_available():
        print("Using GPU, Yay!")
        return torch.device('cuda')
    else:
        print("Using CPU, Meh!")
        return torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    # Wrap a dataloader to move data to device
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


class TIMER:
    def __init__(self, pre=""):
        self.timestmp = time.perf_counter()
        self.pre = pre

    def time(self, label=""):
        if self.timestmp > 0:
            print(f"{self.pre}{label} {(time.perf_counter() - self.timestmp) * 1e3:.0f}ms")
        self.timestmp = time.perf_counter()
