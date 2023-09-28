import torch


def gradient(net, n_grid, device):
    """Evaluate the net on a regular grid and return gradients"""
    # provide input spiketime limits
    early, late = net.input_interval
    xs,ys = torch.meshgrid(torch.linspace(0, 1, steps=n_grid, device=device),
                           torch.linspace(0, 1, steps=n_grid, device=device), 
                           indexing='xy')
    # xs and ys have the y-coord. as their first and the x-coord. as their second index
    xs, ys = xs.flatten().unsqueeze(1), ys.flatten().unsqueeze(1)
    inputs = torch.hstack([xs, ys, 1-xs, 1-ys])

    # have to incorporate early and late time used for YY dataset
    inputs = early + inputs * (late-early)

    label_times, _ = net(inputs)[0]
    n_outputs = label_times.shape[1]
    grads = []
    label_times = label_times.clone().detach().t()
    label_times = label_times.view(n_outputs, n_grid, n_grid)
    for label in label_times:
        grad = torch.gradient(label)
        grad = torch.vstack([direction.unsqueeze(0) for direction in grad]) # grad has now dimensions 2 x n_grid x n_grid
        grads.append(grad)
    gradients = torch.vstack([grad.unsqueeze(0) for grad in grads]) # gradients has dimension n_outputs x 2 x n_grid x n_grid
    return label_times, gradients


def activation_code(net, n_grid, device, short=False):
    """Evaluate the net on a regular grid and return the spikes at each gridpoint"""
    # provide input spiketime limits
    early, late = net.input_interval
    xs,ys = torch.meshgrid(torch.linspace(0, 1, steps=n_grid, device=device),
                        torch.linspace(0, 1, steps=n_grid, device=device), 
                        indexing='xy')
    # xs and ys have the y-coord. as their first and the x-coord. as their second index
    xs, ys = xs.flatten().unsqueeze(1), ys.flatten().unsqueeze(1)
    inputs = torch.hstack([xs, ys, 1-xs, 1-ys])

    # have to incorporate early and late time used for YY dataset
    inputs = early + inputs * (late-early)

    spiketimes, contributions = net(inputs)
    # label_times, hidden_times = spiketimes
    # full, encoded = spikes_to_sparse_code(label_times, hidden_times)
    full, encoded = sparse_code_causal_sets(net, contributions, short=short)
    width = full.shape[1]
    full = full.view(n_grid, n_grid, width)
    encoded = encoded.view(n_grid, n_grid)
    return full, encoded


def spikes_to_code(label_times, hidden_times):
    """Encode the pair of output spiketimes and hidden spiketimes as an integer"""
    spikes_by_layer = []
    spikes_by_layer.append(label_times == torch.inf) 
    for hidden in hidden_times:
        spikes_by_layer.append(hidden == torch.inf)
    full = torch.hstack(spikes_by_layer)
    width = full.shape[1]
    binary_factors = 2.**torch.arange(width)
    binary = (full*binary_factors).sum(dim=1)
    return full, binary

def spikes_to_sparse_code(label_times, hidden_times):
    """Encode the spiketimes of the net with a dictionary"""
    spikes_by_layer = []
    spikes_by_layer.append(label_times == torch.inf) 
    for hidden in hidden_times:
        spikes_by_layer.append(hidden == torch.inf)
    spikes = torch.hstack(spikes_by_layer).int()
    width = spikes.shape[1]

    # encode configurations as strings
    configuration_dict = {}
    n_string = 0
    spikes_as_strings = []
    for spike_vector in spikes:
        spike_string = "".join([str(c.item()) for c in spike_vector])
        if spike_string not in configuration_dict:
            configuration_dict[spike_string] = n_string
            n_string += 1
        spikes_as_strings.append(configuration_dict[spike_string])
    return spikes, torch.tensor(spikes_as_strings)


def sparse_code_causal_sets(net, contributions, short=False):
    """Encode the causal sets of the net
    
    This allows a much finer distinction than just looking at which neurons spike.
    """
    n_data = contributions[0].shape[0]
    contributions_by_layer = list(contributions[1]) + [contributions[0]]
    
    # additional masking required: contributions to spikes which do not lead to outgoing spikes
    # this is done by moving backwards through the layers
    print("before: ", torch.hstack([con.reshape(n_data, -1) for con in contributions_by_layer]).float().mean().data)

    for i in range(1, len(contributions_by_layer)):
        # sum over all outputs to see which inputs are relevant
        relevant_inputs = torch.sum(contributions_by_layer[-i], dim=2).bool()
        # need to ignore possible bias spikes since they are not mapped to by the previous layer
        relevant_inputs_without_bias = relevant_inputs[:, :-len(net.biases[-i])]

        # in the previous layer, only consider the contributions to these relevant outputs
        contributions_by_layer[-i-1] *= relevant_inputs_without_bias.unsqueeze(1)
    
    print("after: ", torch.hstack([con.reshape(n_data, -1) for con in contributions_by_layer]).float().mean().data)

    # determine for each neuron which incoming spikes contribute (empty set if it never spikes)
    contributions_flattened = torch.hstack([con.reshape(n_data, -1) for con in contributions_by_layer])
    # all neurons in the first hidden layer receive the same inputs (only with different weights and delays), could ignore them
    # contributions_flattened = torch.hstack([con.reshape(n_data, -1) for con in contributions_by_layer[1:]]) 

    # encode configurations as strings, using sparsity: encode only the indices of contributing spikes
    configuration_dict = {} 
    n_string = 0
    contributions_as_strings = []
    for contribution_vector in contributions_flattened:
        # for each grid point, get indices where there is a contribution
        contribution_indices = contribution_vector.nonzero()
        contribution_string = "".join([str(c.item()) for c in contribution_indices])
        if short:
            contributions_as_strings.append(len(contribution_string)) # compare the lengths of the encoding strings rather than the strings themselves
        else:
            if contribution_string not in configuration_dict:
                configuration_dict[contribution_string] = n_string
                n_string += 1
            contributions_as_strings.append(configuration_dict[contribution_string])
    print(f"Found {len(set(contributions_as_strings))} linear regions")
    return contributions_flattened, torch.tensor(contributions_as_strings)
