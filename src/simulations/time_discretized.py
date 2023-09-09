import numpy as np
import torch

from simulations.base_simulation import BaseSimulation

torch.set_default_dtype(torch.float64)

class TimeDiscretized(BaseSimulation):
    """Simulate via Euler integration, after each time step compare with a threshold to determine spikes.

    Note that for the linear activation, Euler integration is exact, 
    and inaccuracies are only introduced because spike times are given as multiples of the time step.
    """
    @staticmethod
    def forward(ctx,
                input_spikes, input_weights, input_delays, thresholds,
                sim_params, device, output_times=None):
        """Calculate spike times via integration.

        Arguments:
            ctx: from torch, used for saving for backward pass
            input_spikes, input_weights, input_delays, thresholds: input that is used for calculations
            sim_params: constants used in calculation
            device: torch specifics, for GPU use
            output_times: only used for plotting of the spikes

        Returns:
            output_spikes:  outgoing spike times
        """
        
        n_batch, n_postsyn, delayed_input_spikes = BaseSimulation.prepare_inputs(input_spikes, input_weights, input_delays, thresholds, sim_params)

        # import time
        
        # start = time.time()
        # print("shapes: \t(" + str(n_batch) + ", " + str(n_presyn) + ", " + str(n_postsyn) + ")")

        # stop = time.time()
        # print("delayed inputs: ", (stop-start)*1000) # 1ms, this is negligible
        # start = time.time()
                
        input_times_step = (delayed_input_spikes / sim_params['resolution']).long() # express input spike times as multiples of time steps (here 10ms), shape n_batch x n_input x n_output
        input_times_step[input_times_step > sim_params['steps'] - 1] = sim_params['steps'] - 1 # upper bound on time step values due to finite sim time
        input_times_step[torch.isinf(input_spikes)] = sim_params['steps'] - 1 # might already be covered by previous step
        
        # stop = time.time()
        # print(f'{"time to timestep: ":25}{(stop-start)*1000:.2f}ms') # 1ms, negligible
        # start = time.time()

        # # one-hot code input times for easier multiplication
        # input_transf = torch.zeros(tuple(delayed_input_spikes.shape) + (sim_params['steps'], ),
        #                            device=device, requires_grad=False) # shape: n_batch x n_input x n_output x steps
        # input_transf = torch.eye(sim_params['steps'], device=device)[input_times_step].reshape((n_batch, n_presyn, n_postsyn, sim_params['steps'])) 
        # # identity matrix with row for each time step, evaluated for each input spike:
        # # this yields for every input spike (and corresponding output) a one-hot code for the corresponding time step
        
        # stop = time.time()
        # print(f'{"input transform: ":25}{(stop-start)*1000:.2f}ms') # 80ms
        # start = time.time()
                
        # charge = torch.einsum("abcd,bc->acd", (input_transf, input_weights)) # i.e. multiply each input spike with its weight and sum, represented still as one-hot    
        # # this allows accessing the incoming current (for batch x output) by evaluating only the step number
        
        # stop = time.time()
        # print(f'{"old charge: ":25}{(stop-start)*1000:.2f}ms') # 60ms
        
        stepwise = 'delta' in sim_params # for hat activation function, the stepwise implementation is faster, for the simpler activation function the vectorized one is faster
        plotting = False # TODO: can only be set here? put into config?

        # more efficient calculation of charge, preventing huge one-hot tensor
        # O(n_batch x n_in x n_out), evaluated n_steps times, so in total same complexity as working with full input_transf
        # is still faster since no allocation at once needed
        if not stepwise:
            # start = time.time()
        
            charge = torch.zeros((n_batch, n_postsyn) + (sim_params['steps'],), device=device, requires_grad=False)
            for step in range(sim_params['steps']):
                weighted_rising_spikes = input_weights.unsqueeze(0)*(input_times_step==step) # at the start of the activation function
                charge[:,:,step] = torch.sum(weighted_rising_spikes, dim=1)
                
                if sim_params.get('delta') and sim_params['activation']=='linear': # additional computations for hat function as activation
                    weighted_falling_spikes = input_weights.unsqueeze(0)*(input_times_step==step-sim_params['delta_in_steps']) # at the peak of hat function
                    weighted_end_spikes = input_weights.unsqueeze(0)*(input_times_step==step-2*sim_params['delta_in_steps']) # at the end of hat function
                    charge[:,:,step] -= 2*torch.sum(weighted_falling_spikes, dim=1)
                    charge[:,:,step] += torch.sum(weighted_end_spikes, dim=1)
            
            # stop = time.time()
            # print(f'{"new charge: ":25}{(stop-start)*1000:.2f}ms') # 60ms
        
        if sim_params['activation']=='linear':
            if not stepwise:
                # start = time.time()

                # vectorized implementation for linear activation
                syn = torch.cumsum(charge, dim=2) # synaptic currents are just the cumulative sum of charges over time steps (for each batch and output neuron)
                mem = torch.ones((n_batch, n_postsyn, sim_params['steps']), device=device) * sim_params['leak'] # initialize potential (for batch and output) at its rest value (0)
                mem[:,:,1:] += torch.cumsum(syn[:,:,:-1], dim=2) * sim_params['resolution'] # synaptic current takes only effect in membrane potential at NEXT time step
                # this also means that the last row of syn cannot have effects anymore (makes sense, since the next timestep is not considered anymore)
                # membrane potential is the cumulative sum of current over time steps multiplied by the time step length

                resized_thresholds = thresholds.view(1, n_postsyn) if sim_params.get('train_threshold') else torch.full((1, n_postsyn), sim_params['threshold'])
                
                output_spikes = torch.argmax((mem > resized_thresholds.unsqueeze(-1)).int(), dim=2) * sim_params['resolution'] # get first time step, at which the threshold is crossed
                threshold_not_crossed = torch.max(mem, dim=2)[0] <= resized_thresholds # select the batches and output neurons which do not spike
                output_spikes[threshold_not_crossed] = torch.inf

                if plotting:
                    all_mem = [mem[:,:,i].numpy() for i in range(sim_params['steps'])]

                # stop = time.time()
                # print(f'{"steps vectorized: ":25}{(stop-start)*1000:.2f}ms') # 122ms
            
            else:
                # start = time.time()
                
                # stepwise implementation
                syn = torch.zeros((n_batch, n_postsyn), device=device) # initialize current (for each batch and output) as 0
                mem = torch.ones((n_batch, n_postsyn), device=device) * sim_params['leak'] # initialize potential (for batch and output) at its rest value (0)

                all_mem = []
                output_spikes = torch.ones((n_batch, n_postsyn), device=device) * float('inf') # initialize output spikes as inf

                for step in range(sim_params['steps']): 
                    # compute charge by filtering on spikes within this interval
                    # evaluation O(n_batch x n_in x n_out), summing O(n_in), all done n_steps times
                    charge = torch.zeros((n_batch, n_postsyn), device=device, requires_grad=False)
                    weighted_rising_spikes = input_weights.unsqueeze(0)*(input_times_step==step) # at start of activation function
                    charge += torch.sum(weighted_rising_spikes, dim=1)
                    
                    if sim_params.get('delta'): # additional computations for hat function as activation
                        weighted_falling_spikes = input_weights.unsqueeze(0)*(input_times_step==step-sim_params['delta_in_steps']) # at peak of hat function
                        weighted_end_spikes = input_weights.unsqueeze(0)*(input_times_step==step-2*sim_params['delta_in_steps']) # at end of hat function
                        charge -= 2*torch.sum(weighted_falling_spikes, dim=1)
                        charge += torch.sum(weighted_end_spikes, dim=1)
                    
                    mem += syn * sim_params['resolution'] # update potential by adding the incoming current times the stepsize
                    syn += charge # given a batch and output neuron, the current increases by the corresponding input spikes times weights at this step
                    
                    resized_thresholds = thresholds.unsqueeze(0) if sim_params.get('train_threshold') else torch.full((1, n_postsyn), sim_params['threshold'])
            
                    # mask is a logical_and implemented by multiplication: only update the first time the potential crosses the threshold
                    output_spikes[
                        (torch.isinf(output_spikes) *
                        mem > resized_thresholds)] = step * sim_params['resolution']
                    # reset voltage after spike for plotting
                    # mem[torch.logical_not(torch.isinf(output_spikes))] = 0.
                    if plotting:
                        all_mem.append(mem.numpy())
                
                # stop = time.time()
                # print(f'{"stepwise: ":25}{(stop-start)*1000:.2f}ms') # 122ms
        
        elif sim_params['activation']=='alpha_equaltime':
            syn = torch.zeros((n_batch, n_postsyn), device=device) # initialize current (for each batch and output) as 0
            mem = torch.ones((n_batch, n_postsyn), device=device) * sim_params['leak'] # initialize potential (for batch and output) at its rest value (0)
            
            all_mem = []
            output_spikes = torch.ones((n_batch, n_postsyn), device=device) * float('inf') # initialize output spikes as inf
                
            for step in range(sim_params['steps']): # TODO: vectorize this stepwise computation? probably hard due to threshold etc., but could use triangular matrices
                mem = sim_params['decay_mem'] * (mem - sim_params['leak']) \
                    + 1. / sim_params['g_leak'] * syn * sim_params['resolution'] + sim_params['leak']
                syn = sim_params['decay_syn'] * syn + charge[:, :, step]

                resized_thresholds = thresholds.unsqueeze(0) if sim_params.get('train_threshold') else torch.full((1, n_postsyn), sim_params['threshold'])

                # mask is a logical_and implemented by multiplication: only update the first time the potential crosses the threshold
                output_spikes[
                    (torch.isinf(output_spikes) *
                    mem > resized_thresholds)] = step * sim_params['resolution']
                # reset voltage after spike for plotting
                # mem[torch.logical_not(torch.isinf(output_spikes))] = 0.
                if plotting:
                    all_mem.append(mem.numpy())
        else:
                raise NotImplementedError(f"optimizer {sim_params['activation']} not implemented")

        if plotting:
            import matplotlib.pyplot as plt
            import warnings
            if n_batch >= 9:
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

        # start = time.time()

        if torch.isnan(output_spikes).sum() > 0:
            raise ArithmeticError("There are NaNs in the output times, this means a serious error occured")

        ctx.sim_params = sim_params
        ctx.device = device
        ctx.save_for_backward(
            input_spikes, input_weights, input_delays, thresholds, output_spikes)

        # stop = time.time()
        # print("fifth: ", (stop-start)*1000) # <1ms, only save parameters
            
        return output_spikes