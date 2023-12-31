import numpy as np
import torch

from utils import to_device
from simulations.base_simulation import BaseSimulation

torch.set_default_dtype(torch.float64)

class Explicit(BaseSimulation):
    """Simulate by calculating output spike times explicitly from the input spikes.

    This method comes with no loss in accuracy.
    """
    @staticmethod
    def forward(ctx,
                input_spikes, input_weights, input_delays, thresholds,
                sim_params, device):
        """Calculate spike times using the explicit formula.

        Arguments:
            ctx: from torch, used for saving for backward pass
            input_spikes, input_weights, input_delays, thresholds: input that is used for calculations
            sim_params: constants used in calculation
            device: torch specifics, for GPU use

        Returns:
            output_spikes:  outgoing spike times
        """
        n_batch, _, delayed_input_spikes = BaseSimulation.prepare_inputs(input_spikes, input_weights, input_delays, thresholds, sim_params)

        # create causal set
        # due to to different delays, the sorting has to be done for every output separately
        sort_indices = delayed_input_spikes.argsort(1)
        sorted_spikes = delayed_input_spikes.gather(1, sort_indices)
        sorted_weights = torch.gather(input_weights.unsqueeze(0).expand(n_batch,-1,-1), dim=1, index=sort_indices) # output is of shape n_batch x n_pre x n_post
        # (in each batch) set weights=0 and spiketime=0 for neurons with inf spike time to prevent nans
        mask_of_inf_spikes = torch.isinf(sorted_spikes)
        sorted_spikes_masked = sorted_spikes.clone().detach() # TODO: why detach here? means that these spikes are not part of the comp. graph, i.e. not trained?
        sorted_spikes_masked[mask_of_inf_spikes] = 0.
        sorted_weights[mask_of_inf_spikes] = 0. # TODO: why no clone, detach for weights? (means setting them to 0 is actually part of the comp. graph?)
        
        output_spikes = to_device(torch.ones(input_weights.size()) * np.inf, device) # prepares the array for the output spikes on the device

        if sim_params['activation']=='linear':
            import spiketimes.piecewise_linear as spiketime # use our linear activation model
        elif sim_params['activation']=='alpha_equaltime':
            import spiketimes.alpha_1 as spiketime # use the alpha activation as in Goeltz et al
        elif sim_params['activation']=='alpha_doubletime':
            import spiketimes.alpha_2 as spiketime # use the modified alpha activation as in Goeltz et al
        else:
            raise NotImplementedError(f"optimizer {sim_params['activation']} not implemented")

        tmp_output = spiketime.get_spiketime(
                sorted_spikes_masked,
                sorted_weights,
                thresholds,
                sim_params, device)

        # compare new spike times with previous, to set all cases in which no spike would occur to inf
        before_last_input = tmp_output < sorted_spikes # output cannot happen before the last required input (means we need less inputs)
        after_next_input = tmp_output > sorted_spikes.roll(-1, dims=1) # compare to next input spike, output must happen before (else we need more inputs)
        after_next_input[:, -1, :] = 0.  # last has no subsequent spike, so the output spike is always early enough

        # set non-causal times to inf (so that they are ignored in the min taken to find the output times)
        tmp_output[before_last_input] = float('inf')
        tmp_output[after_next_input] = float('inf')

        # take the earliest computed output spike time, which is the correct one (complexity: n_pre)
        output_spikes, causal_set_lengths = torch.min(tmp_output, dim=1)
        
        # # one could also determine the output spike time requiring the least number of spikes,
        # # namely by finding the index of the first finite outsput spike time along dim 1 (i.e. when adding input spikes one after another)
        # _, causal_set_lengths_new = torch.max(tmp_output<torch.inf, dim=1) 
        # # this is the number of contributing spikes before the output spike happens reduced by 1, i.e. 0 means 1 spike and n means n+1 spikes
        # # the output spike times can then be computed using the number of spikes
        # # (also works if all entries are inf, since in this case the index 0 is returned and the following line correctly yields inf as output time)
        # output_spikes_new = tmp_output.gather(1, causal_set_lengths_new.unsqueeze(1)).squeeze(1)
        # # and the two methods can be compared (results are identical)
        # from torch.testing import assert_close
        # assert_close(output_spikes, output_spikes_new)

        ctx.sim_params = sim_params
        ctx.device = device
        ctx.save_for_backward(
            input_spikes, input_weights, input_delays, thresholds, output_spikes)
        if torch.isnan(output_spikes).sum() > 0:
            raise ArithmeticError("There are NaNs in the output times, this means a serious error occured")
        return output_spikes