import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import torch
import yaml

import utils

torch.set_default_dtype(torch.float64)


def running_mean(x, N=30):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N) # we subtract two parts of the array with a relative shift of N, output has shape of x - N


class Net(torch.nn.Module):
    def __init__(self, network_layout, sim_params, device):
        super(Net, self).__init__()
        self.n_inputs = network_layout['n_inputs']
        self.n_layers = network_layout['n_layers']
        self.layer_sizes = network_layout['layer_sizes']
        self.n_biases = network_layout['n_biases']
        self.weight_means = network_layout['weight_means']
        self.weight_stdevs = network_layout['weight_stdevs']
        
        if network_layout.get('substitute_delays'):
            self.delay_means = network_layout.get('delay_means', [-10.]*self.n_layers)
        else:
            self.delay_means = network_layout.get('delay_means', [0.]*self.n_layers)
        self.delay_stdevs = network_layout.get('delay_stdevs', [0.]*self.n_layers)

        self.threshold_means = network_layout.get('threshold_means', [sim_params['threshold']]*self.n_layers)
        self.threshold_stdevs = network_layout.get('threshold_stdevs', [0.]*self.n_layers)

        self.device = device

        if 'bias_times' in network_layout.keys():
            if len(network_layout['bias_times']) > 0 and isinstance(network_layout['bias_times'][0], (list, np.ndarray)):
                self.bias_times = network_layout['bias_times']
            else:
                self.bias_times = [network_layout['bias_times']] * self.n_layers
        else:
            self.bias_times = []
        self.biases = []
        for i in range(self.n_layers):
            bias = utils.to_device(utils.bias_inputs(self.n_biases[i], self.bias_times[i]), device)
            self.biases.append(bias)

        self.layers = torch.nn.ModuleList()
        layer = utils.EqualtimeLayer(self.n_inputs, self.layer_sizes[0],
                                     sim_params, (self.weight_means[0], self.weight_stdevs[0]),
                                     (self.delay_means[0], self.delay_stdevs[0]),
                                     (self.threshold_means[0], self.threshold_stdevs[0]),
                                     device, self.n_biases[0])
        self.layers.append(layer)
        for i in range(self.n_layers - 1):
            layer = utils.EqualtimeLayer(self.layer_sizes[i], self.layer_sizes[i + 1],
                                         sim_params, (self.weight_means[i + 1], self.weight_stdevs[i + 1]),
                                         (self.delay_means[i+1], self.delay_stdevs[i+1]),
                                         (self.threshold_means[i+1], self.threshold_stdevs[i+1]),
                                         device, self.n_biases[i + 1])
            self.layers.append(layer)

        self.rounding_precision = sim_params.get('rounding_precision')
        self.rounding = self.rounding_precision not in (None, False)
        self.sim_params = sim_params

        if self.rounding:
            print(f"#### Rounding the weights to precision {self.rounding_precision}")
        return

    def clip_weights(self):
        if self.sim_params['clip_weights_max']:
            for i, layer in enumerate(self.layers):
                maxweight = self.sim_params['clip_weights_max']
                self.layers[i].weights.data = torch.clamp(layer.weights.data, -maxweight, maxweight)
        return

    def forward(self, input_times):
        # When rounding we need to save and manipulate weights before forward pass, and after
        if self.rounding and not self.fast_eval:
            float_weights = []
            for layer in self.layers:
                float_weights.append(layer.weights.data)
                layer.weights.data = self.round_weights(layer.weights.data, self.rounding_precision)

        # below, the actual pass through the layers of the network is defined, including the bias terms
        hidden_times = []
        for i in range(self.n_layers):
            input_times_including_bias = torch.cat(
                (input_times,
                    self.biases[i].view(1, -1).expand(len(input_times), -1)),
                1)
            # n_spikes += len(self.biases[i]) leave out number of biases since it is insignificant for the number of spikes
            output_times = self.layers[i](input_times_including_bias)
            if not i == (self.n_layers - 1):
                hidden_times.append(output_times)
                input_times = output_times
            else:
                label_times = output_times
        return_value = label_times, hidden_times

        if self.rounding and not self.fast_eval:
            for layer, floats in zip(self.layers, float_weights):
                layer.weights.data = floats

        return return_value

    def round_weights(self, weights, precision):
        return (weights / precision).round() * precision

    def spike_percentages(self, output_times, hidden_times):
        """
        Compute the ratio of neurons that spike during a forward pass.
        
        Return 
            one ratio for each layer, e.g. 0.5, (1.0,0.5) for one output layer and two hidden ones
        """
        n_output = torch.isfinite(output_times).sum(dim=1).float().mean() # sum over spikes and average over batches
        n_hidden = [torch.isfinite(hidden).sum(dim=1).float().mean() for hidden in hidden_times]
        output_percentage = n_output / self.layer_sizes[-1]
        hidden_percentages = [n_hidden[i] / self.layer_sizes[-2-i] for i in range(self.n_layers-1)]
        return output_percentage, hidden_percentages


def load_data(dirname, filename, dataname):
    path = dirname + '/' + filename + dataname
    data = np.load(path, allow_pickle=True)
    return data


def load_config(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    return data['dataset'], data['neuron_params'], data['network_layout'], data['training_params']


def validation_step(net, criterion, loader, device, return_input=False):
    all_outputs = []
    all_labels = []
    all_inputs = []
    with torch.no_grad():
        losses = []
        num_correct = 0
        num_shown = 0
        for j, data in enumerate(loader):
            inputs, labels = data
            input_times = utils.to_device(inputs.clone().type(torch.float64), device)
            outputs, _ = net(input_times)
            selected_classes = criterion.select_classes(outputs)
            num_correct += len(outputs[selected_classes == labels])
            num_shown += len(labels)
            loss = criterion(outputs, labels) * len(labels)
            losses.append(loss)
            all_outputs.append(outputs)
            all_labels.append(labels)
            if return_input:
                all_inputs.append(input_times)

        loss = sum(losses) / float(num_shown)  # can't use simple mean because batches might have diff size
        accuracy = float(num_correct) / num_shown
        # flatten output and label lists
        outputs = [item for sublist in all_outputs for item in sublist]
        labels = [item.item() for sublist in all_labels for item in sublist]
        if return_input:
            inputs = (torch.stack([item for sublist in all_inputs for item in sublist])).detach().cpu().numpy()
            return loss, accuracy, outputs, labels, inputs
        else:
            return loss, accuracy, outputs, labels, None


def check_bump_weights(net, hidden_times, label_times, training_params, epoch, batch, bump_val, last_weights_bumped):
    """determines if spikes were lost, adapts bump_val and bumps weights

    only foremost layer gets bumped: if in an earlier layer spikes are missing,
    chances are that in subsequent layers there will be too little input and missing
    spikes as well.
    return value weights_bumped:
        positive integer is hidden id that needed bump,
        -1: label layer needed bump
        -2: no bumping needed
    """
    weights_bumped = -2
    for i, times in enumerate(hidden_times):
        # we want mean over batches and neurons
        denominator = times.shape[0] * times.shape[1]
        non_spikes = torch.isinf(times) + torch.isnan(times)
        num_nonspikes = float(non_spikes.bool().sum())
        if num_nonspikes / denominator > training_params['max_num_missing_spikes'][i]:
            weights_bumped = i
            break
    else:
        # else after for only executed if no break happened
        i = -1
        denominator = label_times.shape[0] * label_times.shape[1]
        non_spikes = torch.isinf(label_times) + torch.isnan(label_times)
        num_nonspikes = float(non_spikes.bool().sum())
        if num_nonspikes / denominator > training_params['max_num_missing_spikes'][-1]:
            weights_bumped = -1
    if weights_bumped != -2:
        if training_params['weight_bumping_exp'] and weights_bumped == last_weights_bumped:
            bump_val *= 2
        else:
            bump_val = training_params['weight_bumping_value']
        if training_params['weight_bumping_targeted']:
            # make bool and then int to have either zero or ones
            should_bump = non_spikes.sum(axis=0).bool().int()
            n_in = net.layers[i].weights.data.size()[0]
            bumps = should_bump.repeat(n_in, 1) * bump_val
            net.layers[i].weights.data += bumps
        else:
            net.layers[i].weights.data += bump_val

        # print("epoch {0}, batch {1}: missing {4} spikes, bumping weights by {2} (targeted_bump={3})".format(
        #     epoch, batch, bump_val, training_params['weight_bumping_targeted'],
        #     "label" if weights_bumped == -1 else "hidden"))
    return weights_bumped, bump_val


def save_untrained_network(dirname, filename, net):
    if (dirname is None) or (filename is None):
        return
    path = '../experiment_results/' + dirname
    try:
        os.makedirs(path)
        print("Directory ", path, " Created ")
    except FileExistsError:
        print("Directory ", path, " already exists")
    if not path[-1] == '/':
        path += '/'
    # save network
    torch.save(net, path + filename + '_untrained_network.pt')
    return


def save_config(dirname, filename, neuron_params, network_layout, training_params, epoch_dir=(False, -1)):
    if (dirname is None) or (filename is None):
        return
    dirname = '../experiment_results/' + dirname
    if not dirname[-1] == '/':
        dirname += '/'
    if epoch_dir[0]:
        dirname += 'epoch_{}/'.format(epoch_dir[1])
    if not osp.isdir(dirname):
        os.makedirs(dirname)
        print("Directory ", dirname, " Created ")
    # save parameter configs
    with open(osp.join(dirname, 'config.yaml'), 'w') as f:
        yaml.dump({"dataset": filename, "neuron_params": neuron_params,
                   "network_layout": network_layout, "training_params": training_params}, f)
    return


def save_data(dirname, filename, net, all_parameters, train_losses, train_accuracies, val_losses, val_accuracies,
              val_labels, mean_val_outputs_sorted, std_val_outputs_sorted, spike_percentages, epoch_dir=(False, -1)):
    if (dirname is None) or (filename is None):
        return
    dirname = '../experiment_results/' + dirname
    if not dirname[-1] == '/':
        dirname += '/'
    if epoch_dir[0]:
        dirname += 'epoch_{}/'.format(epoch_dir[1])
    try:
        os.makedirs(dirname)
        print("Directory ", dirname, " Created ")
    except FileExistsError:
        print("Directory ", dirname, " already exists")
    # save network
    torch.save(net, dirname + filename + '_network.pt')

    # save training result
    for key in ["label_weights", "hidden_weights", "label_delays", "hidden_delays", "label_thresholds", "hidden_thresholds"]:
        np.save(dirname + filename + '_{}_training.npy'.format(key), all_parameters[key])
    np.save(dirname + filename + '_train_losses.npy', train_losses)
    np.save(dirname + filename + '_train_accuracies.npy', train_accuracies)
    np.save(dirname + filename + '_val_losses.npy', val_losses)
    np.save(dirname + filename + '_val_accuracies.npy', val_accuracies)
    np.save(dirname + filename + '_val_labels.npy', val_labels)
    np.save(dirname + filename + '_mean_val_outputs_sorted.npy', mean_val_outputs_sorted)
    np.save(dirname + filename + '_std_val_outputs_sorted.npy', std_val_outputs_sorted)
    np.save(dirname + filename + '_spike_percentages.npy', spike_percentages)
    return

def save_loss_plot(training_progress, trainloader, path=None, spike_percentages=None):
    save_path = 'live_accuracy.png'
    if path:
        save_path = os.path.join(path, 'live_accuracy.png')
    fig, ax = plt.subplots(1, 1)
    tmp = 1. - running_mean(training_progress, N=30)
    ax.plot(np.arange(len(tmp)) / len(trainloader), tmp, c="k", label="error")
    ax.set_ylim(0.005, 1.0)
    ax.set_yscale('log')
    ax.set_xlabel("epochs")
    ax.set_ylabel("error [%] (running_mean of 30 batches)")
    ax.axhline(0.30, c="grey")
    ax.axhline(0.05, c="grey")
    ax.axhline(0.01, c="grey")
    ax.set_yticks([0.01, 0.05, 0.1, 0.3])
    ax.set_yticklabels([1, 5, 10, 30])
    if spike_percentages:
        ax2 = ax.twinx()
        output_percentage = running_mean(spike_percentages[0], N=30)
        n_layers = len(spike_percentages)
        for i in range(n_layers-1):
            hidden_percentage = running_mean(spike_percentages[-1-i], N=30)
            ax2.plot(np.arange(len(hidden_percentage)) / len(trainloader), hidden_percentage, label=f'layer {1+i} spike ratio')
        ax2.plot(np.arange(len(output_percentage)) / len(trainloader), output_percentage, label=f"layer {n_layers} spike ratio")
        # ax2.spines['right'].set_color('grey')
        ax2.set_ylabel("neuron spike ratio (running mean of 30 batches)")
        ax2.set_ylim(0.45, 1.05)
    fig.legend(framealpha=1, loc="upper center")
    
    fig.savefig(save_path)
    print("===========Saved live accuracy plot")
    plt.close(fig)


def save_result_spikes(dirname, filename, train_times, train_labels, train_inputs,
                       test_times, test_labels, test_inputs, epoch_dir=(False, -1)):
    if (dirname is None) or (filename is None):
        return
    dirname = '../experiment_results/' + dirname
    if not dirname[-1] == '/':
        dirname += '/'
    if epoch_dir[0]:
        dirname += 'epoch_{}/'.format(epoch_dir[1])
    # stunt to avoid saving tensors
    train_times = np.array([item.detach().cpu().numpy() for item in train_times])
    test_times = np.array([item.detach().cpu().numpy() for item in test_times])
    np.save(dirname + filename + '_train_spiketimes.npy', train_times)
    np.save(dirname + filename + '_train_labels.npy', train_labels)
    if train_inputs is not None:
        np.save(dirname + filename + '_train_inputs.npy', train_inputs)
    np.save(dirname + filename + '_test_spiketimes.npy', test_times)
    np.save(dirname + filename + '_test_labels.npy', test_labels)
    if test_inputs is not None:
        np.save(dirname + filename + '_test_inputs.npy', test_inputs)
    return


def save_optim_state(dirname, filename, optimizer, scheduler, np_rand_state, torch_rand_state, epoch_dir=(False, -1)):
    if (dirname is None) or (filename is None):
        return
    dirname = '../experiment_results/' + dirname
    if not dirname[-1] == '/':
        dirname += '/'
    if epoch_dir[0]:
        dirname += 'epoch_{}/'.format(epoch_dir[1])
    with open(dirname + filename + '_optim_state.yaml', 'w') as f:
        yaml.dump([optimizer.state_dict(), scheduler.state_dict()], f)
    # if saving a snapshot, save state of rngs
    torch.save(torch_rand_state, dirname + filename + '_torch_rand_state.pt')
    numpy_dict = {'first': np_rand_state[0],
                  'second': np_rand_state[1],
                  'third': np_rand_state[2],
                  }
    with open(dirname + filename + '_numpy_rand_state.yaml', 'w') as f:
        yaml.dump([numpy_dict], f)
    return


def setup_lr_scheduling(params, optimizer):
    if params['type'] is None:
        return None
    elif params['type'] == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=params['step_size'],
                                               gamma=params['gamma'])
    elif params['type'] == 'MultiStepLR':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=params['milestones'],
                                                    gamma=params['gamma'])
    elif params['type'] == 'ExponentialLR':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    gamma=params['gamma'])
    else:
        raise IOError('WARNING: Chosen scheduler unknown. Use StepLR or MultiStepLR or ExponentialLR')


def load_optim_state(dirname, filename, net, training_params):
    path = dirname + '/' + filename + '_optim_state.yaml'
    with open(path) as f:
        data = yaml.load_all(f, Loader=yaml.Loader)
        all_configs = next(iter(data))
    if training_params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=training_params['learning_rate'])
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=training_params['learning_rate'],
                                    momentum=training_params['momentum'])
    optim_state = all_configs[0]
    optimizer.load_state_dict(optim_state)
    scheduler = setup_lr_scheduling(training_params['lr_scheduler'], optimizer)
    schedule_state = all_configs[1]
    scheduler.load_state_dict(schedule_state)
    path = dirname + '/' + filename + '_numpy_rand_state.yaml'
    try:
        with open(path) as f:
            data = yaml.load_all(f, Loader=yaml.Loader)
            all_configs = next(iter(data))
        numpy_dict = all_configs[0]
        numpy_rand_state = (numpy_dict['first'], numpy_dict['second'], numpy_dict['third'])
    except IOError:
        numpy_rand_state = None
    path = dirname + '/' + filename + '_torch_rand_state.pt'
    try:
        torch_rand_state = torch.load(path)
    except IOError:
        torch_rand_state = None
    return optimizer, scheduler, torch_rand_state, numpy_rand_state


def apply_noise(input_times, noise_params, device):
    shape = input_times.size()
    noise = utils.to_device(torch.zeros(shape), device)
    noise.normal_(noise_params['mean'], noise_params['std_dev'])
    input_times = input_times + noise
    negative = input_times < 0.
    input_times[negative] *= -1
    return input_times


def run_epochs(e_start, e_end, net, criterion, optimizer, scheduler, device, trainloader, valloader,
               num_classes, all_parameters, all_train_loss, all_validate_loss, std_validate_outputs_sorted,
               mean_validate_outputs_sorted, tmp_training_progress, all_validate_accuracy,
               all_train_accuracy, weight_bumping_steps, spike_percentages, training_params):
    bump_val = training_params['weight_bumping_value']
    last_weights_bumped = -2  # means no bumping happened last time
    last_learning_rate = 0  # for printing learning rate at beginning
    noisy_training = training_params.get('training_noise') not in (False, None)
    print_step = max(1, int(training_params['epoch_number'] * training_params['print_step_percent'] / 100.))

    eval_times = []
    for epoch in range(e_start, e_end, 1):
        train_loss = []
        num_correct = 0
        num_shown = 0
        for j, data in enumerate(trainloader):
            inputs, labels = data
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs, dtype=torch.float64)
            if not inputs.dtype == torch.float64:
                inputs = inputs.double()
            input_times = utils.to_device(inputs, device)
            if noisy_training:
                input_times = apply_noise(input_times, training_params['training_noise'], device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            import time 
            start = time.time()
            label_times, hidden_times = net(input_times)
            stop = time.time()
            #print(f'eval of net: {(stop-start)*1000:.1f}')
            eval_times.append((stop-start)*1000)
            #print(f'mean eval time: {sum(eval_times)/len(eval_times):.2f}ms')
            #print(f'n eval times: {len(eval_times):4}')
            output_percentage, hidden_percentages = net.spike_percentages(label_times, hidden_times)
            selected_classes = criterion.select_classes(label_times)
            # Either do the backward pass or bump weights because spikes are missing
            last_weights_bumped, bump_val = check_bump_weights(net, hidden_times, label_times,
                                                               training_params, epoch, j, bump_val, last_weights_bumped)
            live_plot = True
            if last_weights_bumped != -2:  # means bumping happened
                weight_bumping_steps.append(epoch * len(trainloader) + j)
            else:
                loss = criterion(label_times, labels)
                loss.backward()
                optimizer.step()
                # on hardware we need extra step to write weights
                train_loss.append(loss.item())
            net.clip_weights()
            num_correct += len(label_times[selected_classes == labels])
            num_shown += len(labels)
            tmp_training_progress.append(len(label_times[selected_classes == labels]) / len(labels))
            spike_percentages[0].append(output_percentage)
            [spike_percentages[1+i].append(hidden_percentages[i]) for i in range(net.n_layers-1)]


            if j % 100 == 0:
                [print(f'ratio of hidden spikes in layer {i}: {hidden_percentages[-i]:.3f}') for i in range(1,net.n_layers)]
                print(f'ratio of output spikes in layer {net.n_layers}: {output_percentage:.3f}')
                if live_plot:
                    save_loss_plot(tmp_training_progress, trainloader, spike_percentages=spike_percentages)

        if len(train_loss) > 0:
            all_train_loss.append(np.mean(train_loss))
        else:
            all_train_loss.append(np.nan)
        train_accuracy = num_correct / num_shown if num_shown > 0 else np.nan
        all_train_accuracy.append(train_accuracy)
        if scheduler is not None:
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            if not (last_learning_rate == lr):
                print('setting learning_rate to {0:.5f}'.format(lr))
            last_learning_rate = lr

        # evaluate on validation set
        with torch.no_grad():
            validate_loss, validate_accuracy, validate_outputs, validate_labels, _ = validation_step(
                net, criterion, valloader, device)

            tmp_class_outputs = [[] for i in range(num_classes)]
            for pattern in range(len(validate_outputs)):
                true_label = validate_labels[pattern]
                tmp_class_outputs[int(true_label)].append(validate_outputs[pattern].cpu().detach().numpy())
            for i in range(num_classes):
                tmp_times = np.array(tmp_class_outputs[i])
                tmp_times[np.isinf(tmp_times)] = np.NaN
                mask_notAllNan = np.logical_not(np.isnan(tmp_times)).sum(0) > 0
                mean_times = np.ones(tmp_times.shape[1:]) * np.NaN
                std_times = np.ones(tmp_times.shape[1:]) * np.NaN
                mean_times[mask_notAllNan] = np.nanmean(tmp_times[:, mask_notAllNan], 0)
                std_times[mask_notAllNan] = np.nanstd(tmp_times[:, mask_notAllNan], 0)
                mean_validate_outputs_sorted[i].append(mean_times)
                std_validate_outputs_sorted[i].append(std_times)

            all_validate_accuracy.append(validate_accuracy)
            all_parameters["label_weights"].append(net.layers[-1].weights.data.cpu().detach().numpy().copy())
            [all_parameters["hidden_weights"].append(net.layers[-2-i].weights.data.cpu().detach().numpy().copy()) for i in range(net.n_layers-1)]
            if training_params["train_delay"]:
                all_parameters["label_delays"].append(net.layers[-1].delays.data.cpu().detach().numpy().copy())
                [all_parameters["hidden_delays"].append(net.layers[-2-i].delays.data.cpu().detach().numpy().copy()) for i in range(net.n_layers-1)]
            if training_params["train_threshold"]:
                all_parameters["label_thresholds"].append(net.layers[-1].thresholds.data.cpu().detach().numpy().copy())
                [all_parameters["hidden_thresholds"].append(net.layers[-2-i].thresholds.data.cpu().detach().numpy().copy()) for i in range(net.n_layers-1)]
            all_validate_loss.append(validate_loss.data.cpu().detach().numpy())

        if (epoch % print_step) == 0:
            print("... {0:.1f}% done, train accuracy: {4:.3f}, validation accuracy: {1:.3f}, "
                  "trainings loss: {2:.5f}, validation loss: {3:.5f}".format(
                      epoch * 100 / training_params['epoch_number'], validate_accuracy,
                      np.mean(train_loss) if len(train_loss) > 0 else np.NaN,
                      validate_loss, train_accuracy),
                  flush=True)

        result_dict = {'all_parameters': all_parameters,
                       'all_train_loss': all_train_loss,
                       'all_validate_loss': all_validate_loss,
                       'std_validate_outputs_sorted': std_validate_outputs_sorted,
                       'mean_validate_outputs_sorted': mean_validate_outputs_sorted,
                       'tmp_training_progress': tmp_training_progress,
                       'all_validate_accuracy': all_validate_accuracy,
                       'all_train_accuracy': all_train_accuracy,
                       'weight_bumping_steps': weight_bumping_steps,
                       'spike_percentages': spike_percentages
                       }
    return net, criterion, optimizer, scheduler, result_dict


def train(training_params, network_layout, neuron_params, dataset_train, dataset_val, dataset_test,
          foldername='tmp', filename=''):
    if not training_params['torch_seed'] is None:
        torch.manual_seed(training_params['torch_seed'])
    if not training_params['numpy_seed'] is None:
        np.random.seed(training_params['numpy_seed'])
    if 'optimizer' not in training_params.keys():
        training_params['optimizer'] = 'sgd'
    if 'enforce_cpu' in training_params.keys() and training_params['enforce_cpu']:
        device = torch.device('cpu')
    elif 'enforce_mps' in training_params.keys() and training_params['enforce_mps']:
        device = torch.device('mps')
    else:
        device = utils.get_default_device()
    if not device in ['cpu', 'mps']:
        torch.cuda.manual_seed(training_params['torch_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if not isinstance(training_params['max_num_missing_spikes'], (list, tuple, np.ndarray)):
        training_params['max_num_missing_spikes'] = [
            training_params['max_num_missing_spikes']] * network_layout['n_layers']

    # save parameter config
    save_config(foldername, filename, neuron_params, network_layout, training_params)

    # create sim params
    sim_params = {k: training_params.get(k, False)
                  for k in ['use_forward_integrator', 'resolution', 'sim_time',
                            'rounding_precision', 'max_dw_norm',
                            'clip_weights_max', 'train_delay', 'train_threshold', 'substitute_delay']
                  }
    sim_params.update(neuron_params)

    print('training_params')
    print(training_params)
    print('network_layout')
    print(network_layout)
    print('using optimizer {0}'.format(training_params['optimizer']))

    # setup saving of snapshots
    savepoints = training_params.get('epoch_snapshots', [])
    if not training_params['epoch_number'] in savepoints:
        savepoints.append(training_params['epoch_number'])
    print('Saving snapshots at epochs {}'.format(savepoints))

    print("loading data")
    loader_train = utils.DeviceDataLoader(torch.utils.data.DataLoader(
        dataset_train, batch_size=training_params['batch_size'], shuffle=True), device)
    loader_val = utils.DeviceDataLoader(torch.utils.data.DataLoader(
        dataset_val, batch_size=training_params.get('batch_size_eval', None), shuffle=False), device)
    loader_test = utils.DeviceDataLoader(torch.utils.data.DataLoader(
        dataset_test, batch_size=training_params.get('batch_size_eval', None), shuffle=False), device)

    print("generating network")
    net = utils.to_device(
        Net(network_layout, sim_params, device),
        device)
    save_untrained_network(foldername, filename, net)

    print("loss function")
    criterion = utils.GetLoss(training_params, 
                              network_layout['layer_sizes'][-1],
                              sim_params['tau_syn'], device)

    if training_params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=training_params['learning_rate'])
    elif training_params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=training_params['learning_rate'],
                                    momentum=training_params['momentum'])
    else:
        raise NotImplementedError(f"optimizer {training_params['optimizer']} not implemented")
    scheduler = None
    if 'lr_scheduler' in training_params.keys():
        scheduler = setup_lr_scheduling(training_params['lr_scheduler'], optimizer)

    # evaluate on validation set before training
    num_classes = network_layout['layer_sizes'][-1]
    all_parameters = {"label_weights": [], "hidden_weights": [], "label_delays": [], "hidden_delays": [], "label_thresholds": [], "hidden_thresholds": []}
    all_train_loss = []
    all_validate_loss = []
    std_validate_outputs_sorted = [[] for i in range(num_classes)]
    mean_validate_outputs_sorted = [[] for i in range(num_classes)]
    tmp_training_progress = []
    all_validate_accuracy = []
    all_train_accuracy = []
    weight_bumping_steps = []
    spike_percentages = [[] for i in range(net.n_layers)]
    print("initial validation started")
    with torch.no_grad():
        loss, validate_accuracy, validate_outputs, validate_labels, _ = validation_step(
            net, criterion, loader_val, device)
        tmp_class_outputs = [[] for i in range(num_classes)]
        for pattern in range(len(validate_outputs)):
            true_label = validate_labels[pattern]
            tmp_class_outputs[int(true_label)].append(validate_outputs[pattern].cpu().detach().numpy())
        for i in range(num_classes):
            tmp_times = np.array(tmp_class_outputs[i])
            inf_mask = np.isinf(tmp_times)
            tmp_times[inf_mask] = np.NaN
            mean_times = np.nanmean(tmp_times, 0)
            std_times = np.nanstd(tmp_times, 0)
            mean_validate_outputs_sorted[i].append(mean_times)
            std_validate_outputs_sorted[i].append(std_times)
        print('Initial validation accuracy: {:.3f}'.format(validate_accuracy))
        print('Initial validation loss: {:.3f}'.format(loss))
        all_validate_accuracy.append(validate_accuracy)
        all_parameters["label_weights"].append(net.layers[-1].weights.data.cpu().detach().numpy().copy())
        [all_parameters["hidden_weights"].append(net.layers[-2-i].weights.data.cpu().detach().numpy().copy()) for i in range(net.n_layers-1)]
        if training_params["train_delay"]:
            all_parameters["label_delays"].append(net.layers[-1].delays.data.cpu().detach().numpy().copy())
            [all_parameters["hidden_delays"].append(net.layers[-2-i].delays.data.cpu().detach().numpy().copy()) for i in range(net.n_layers-1)]
        if training_params["train_threshold"]:
            all_parameters["label_thresholds"].append(net.layers[-1].thresholds.data.cpu().detach().numpy().copy())
            [all_parameters["hidden_thresholds"].append(net.layers[-2-i].thresholds.data.cpu().detach().numpy().copy()) for i in range(net.n_layers-1)]
        all_validate_loss.append(loss.data.cpu().detach().numpy())

    print("training started")
    for i, e_end in enumerate(savepoints):
        if i == 0:
            e_start = 0
        else:
            e_start = savepoints[i - 1]
        print('Starting training from epoch {0} to epoch {1}'.format(e_start, e_end))
        net, criterion, optimizer, scheduler, result_dict = run_epochs(
            e_start, e_end, net, criterion,
            optimizer, scheduler, device, loader_train,
            loader_val, num_classes, all_parameters,
            all_train_loss, all_validate_loss,
            std_validate_outputs_sorted,
            mean_validate_outputs_sorted,
            tmp_training_progress, all_validate_accuracy,
            all_train_accuracy, weight_bumping_steps, 
            spike_percentages, training_params)
        print('Ending training from epoch {0} to epoch {1}'.format(e_start, e_end))
        all_parameters = result_dict['all_parameters']
        all_train_loss = result_dict['all_train_loss']
        all_validate_loss = result_dict['all_validate_loss']
        std_validate_outputs_sorted = result_dict['std_validate_outputs_sorted']
        mean_validate_outputs_sorted = result_dict['mean_validate_outputs_sorted']
        all_validate_accuracy = result_dict['all_validate_accuracy']
        all_train_accuracy = result_dict['all_train_accuracy']
        weight_bumping_steps = result_dict['weight_bumping_steps']
        tmp_training_progress = result_dict['tmp_training_progress']
        spike_percentages = result_dict['spike_percentages']
        save_data(foldername, filename, net, all_parameters, all_train_loss,
                  all_train_accuracy, all_validate_loss, all_validate_accuracy,
                  validate_labels, mean_validate_outputs_sorted, std_validate_outputs_sorted,
                  spike_percentages, epoch_dir=(True, e_end))
        # also save the loss curve in the results folder
        if foldername:
            loss_plot_path = osp.join('../experiment_results', foldername, 'epoch_{}/'.format(e_end))
            save_loss_plot(tmp_training_progress, loader_train, path=loss_plot_path, spike_percentages=spike_percentages)

        # evaluate on test set
        return_input = False
        # run again on training set (for spiketime saving)
        loss, final_train_accuracy, final_train_outputs, final_train_labels, final_train_inputs = validation_step(
            net, criterion, loader_train, device, return_input=return_input)
        loss, test_accuracy, test_outputs, test_labels, test_inputs = validation_step(
            net, criterion, loader_test, device, return_input=return_input)

        save_result_spikes(foldername, filename, final_train_outputs, final_train_labels, final_train_inputs,
                           test_outputs, test_labels, test_inputs, epoch_dir=(True, e_end))

        # each savepoint needs config to be able to run inference for eval
        save_config(foldername, filename, neuron_params, network_layout, training_params, epoch_dir=(True, e_end))
        numpy_rand_state = np.random.get_state()
        torch_rand_state = torch.get_rng_state()
        save_optim_state(foldername, filename, optimizer, scheduler, numpy_rand_state,
                         torch_rand_state, epoch_dir=(True, e_end))
    print("Training finished")
    print('####################')
    print('Test accuracy: {}'.format(test_accuracy))

    save_data(foldername, filename, net, all_parameters, all_train_loss,
              all_train_accuracy, all_validate_loss, all_validate_accuracy,
              validate_labels, mean_validate_outputs_sorted, std_validate_outputs_sorted, spike_percentages)

    # also save the loss curve in the results folder
    if foldername:
        loss_plot_path = os.path.join('../experiment_results', foldername)
        save_loss_plot(tmp_training_progress, loader_train, path=loss_plot_path, spike_percentages=spike_percentages)
    return net


def continue_training(dirname, filename, start_epoch, savepoints, dataset_train, dataset_val, dataset_test,
                      net=None):
    dirname_long = dirname + '/epoch_{}/'.format(start_epoch)
    dataset, neuron_params, network_layout, training_params = load_config(osp.join(dirname_long, "config.yaml"))
    if not training_params['torch_seed'] is None:
        torch.manual_seed(training_params['torch_seed'])
    if not training_params['numpy_seed'] is None:
        np.random.seed(training_params['numpy_seed'])
    weight_bumping_steps = []
    tmp_training_progress = []
    spike_percentages = list(load_data(dirname_long, filename, '_spike_percentages.npy'))
    all_train_loss = list(load_data(dirname_long, filename, '_train_losses.npy'))
    all_train_accuracy = list(load_data(dirname_long, filename, '_train_accuracies.npy'))
    all_validate_loss = list(load_data(dirname_long, filename, '_val_losses.npy'))
    all_validate_accuracy = list(load_data(dirname_long, filename, '_val_accuracies.npy'))
    all_parameters = {
        key: list(load_data(dirname_long, filename, '_{}_training.npy'.format(key))) \
            for key in ["label_weights", "hidden_weights", "label_delays", "hidden_delays", "label_thresholds", "hidden_thresholds"] 
    }
    mean_validate_outputs_sorted = list(load_data(dirname_long, filename, '_mean_val_outputs_sorted.npy'))
    mean_validate_outputs_sorted = [list(item) for item in mean_validate_outputs_sorted]
    std_validate_outputs_sorted = list(load_data(dirname_long, filename, '_std_val_outputs_sorted.npy'))
    std_validate_outputs_sorted = [list(item) for item in std_validate_outputs_sorted]

    if 'enforce_cpu' in training_params.keys() and training_params['enforce_cpu']:
        device = torch.device('cpu')
    elif 'enforce_mps' in training_params.keys() and training_params['enforce_mps']:
        device = torch.device('mps')
    else:
        device = utils.get_default_device()
    if not device in ['cpu', 'mps']:
        torch.cuda.manual_seed(training_params['torch_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if not isinstance(training_params['max_num_missing_spikes'], (list, tuple, np.ndarray)):
        training_params['max_num_missing_spikes'] = [
            training_params['max_num_missing_spikes']] * network_layout['n_layers']

    # create sim params
    sim_params = {k: training_params.get(k, False)
                  for k in ['use_forward_integrator', 'resolution', 'sim_time',
                            'rounding_precision', 'max_dw_norm',
                            'clip_weights_max', 'train_delay', 'train_threshold', 'substitute_delay']
                  }
    sim_params.update(neuron_params)

    print('training_params')
    print(training_params)
    print('network_layout')
    print(network_layout)

    # setup saving of snapshots
    print('Saving snapshots at epochs {}'.format(savepoints))

    print("loading data")
    loader_train = utils.DeviceDataLoader(torch.utils.data.DataLoader(
        dataset_train, batch_size=training_params['batch_size'], shuffle=True), device)
    loader_val = utils.DeviceDataLoader(torch.utils.data.DataLoader(
        dataset_val, batch_size=training_params.get('batch_size_eval', None), shuffle=False), device)
    loader_test = utils.DeviceDataLoader(torch.utils.data.DataLoader(
        dataset_test, batch_size=training_params.get('batch_size_eval', None), shuffle=False), device)

    if net is None:
        print("loading network")
        net = utils.network_load(dirname_long, filename, device)
    else:
        print("reusing network")
    if len(savepoints) == 1 and savepoints[0] == start_epoch:
        print("not doing anything with net, only returning it")
        return net

    print("loading optimizer and scheduler")
    criterion = utils.GetLoss(training_params,
                              network_layout['layer_sizes'][-1],
                              sim_params['tau_syn'], device)
    optimizer, scheduler, torch_rand_state, numpy_rand_state = load_optim_state(
        dirname_long, filename, net, training_params)

    # evaluate on validation set before training
    num_classes = network_layout['layer_sizes'][-1]

    print("initial validation started")
    with torch.no_grad():
        loss, validate_accuracy, validate_outputs, validate_labels, _ = validation_step(
            net, criterion, loader_val, device)
        tmp_class_outputs = [[] for i in range(num_classes)]
        for pattern in range(len(validate_outputs)):
            true_label = validate_labels[pattern]
            tmp_class_outputs[true_label].append(validate_outputs[pattern].cpu().detach().numpy())
        for i in range(num_classes):
            tmp_times = np.array(tmp_class_outputs[i])
            inf_mask = np.isinf(tmp_times)
            tmp_times[inf_mask] = np.NaN
            mean_times = np.nanmean(tmp_times, 0)
            std_times = np.nanstd(tmp_times, 0)
            mean_validate_outputs_sorted[i].append(mean_times)
            std_validate_outputs_sorted[i].append(std_times)
        print('Initial validation accuracy: {:.3f}'.format(validate_accuracy))
        print('Initial validation loss: {:.3f}'.format(loss))
        all_validate_accuracy.append(validate_accuracy)
        all_parameters["label_weights"].append(net.layers[-1].weights.data.cpu().detach().numpy().copy())
        [all_parameters["hidden_weights"].append(net.layers[-2-i].weights.data.cpu().detach().numpy().copy()) for i in range(net.n_layers-1)]
        if training_params["train_delay"]:
            all_parameters["label_delays"].append(net.layers[-1].delays.data.cpu().detach().numpy().copy())
            [all_parameters["hidden_delays"].append(net.layers[-2-i].delays.data.cpu().detach().numpy().copy()) for i in range(net.n_layers-1)]
        if training_params["train_threshold"]:
            all_parameters["label_thresholds"].append(net.layers[-1].thresholds.data.cpu().detach().numpy().copy())
            [all_parameters["hidden_thresholds"].append(net.layers[-2-i].thresholds.data.cpu().detach().numpy().copy()) for i in range(net.n_layers-1)]
        all_validate_loss.append(loss.data.cpu().detach().numpy())

    # only seed after initial validation run
    if torch_rand_state is None:
        print("WARNING: Could not load torch rand state, will carry on without")
    else:
        torch.set_rng_state(torch_rand_state)
    if numpy_rand_state is None:
        print("WARNING: Could not load numpy rand state, will carry on without")
    else:
        np.random.set_state(numpy_rand_state)
    print("training started")
    for i, e_end in enumerate(savepoints):
        if i == 0:
            e_start = start_epoch
        else:
            e_start = savepoints[i - 1]
        print('Starting training from epoch {0} to epoch {1}'.format(e_start, e_end))
        net, criterion, optimizer, scheduler, result_dict = run_epochs(
            e_start, e_end, net, criterion,
            optimizer, scheduler, device, loader_train,
            loader_val, num_classes, all_parameters,
            all_train_loss, all_validate_loss,
            std_validate_outputs_sorted,
            mean_validate_outputs_sorted,
            tmp_training_progress, all_validate_accuracy,
            all_train_accuracy, weight_bumping_steps,
            spike_percentages, training_params)
        print('Ending training from epoch {0} to epoch {1}'.format(e_start, e_end))
        all_parameters = result_dict['all_parameters']
        all_train_loss = result_dict['all_train_loss']
        all_validate_loss = result_dict['all_validate_loss']
        std_validate_outputs_sorted = result_dict['std_validate_outputs_sorted']
        mean_validate_outputs_sorted = result_dict['mean_validate_outputs_sorted']
        all_validate_accuracy = result_dict['all_validate_accuracy']
        all_train_accuracy = result_dict['all_train_accuracy']
        weight_bumping_steps = result_dict['weight_bumping_steps']
        tmp_training_progress = result_dict['tmp_training_progress']
        spike_percentages = result_dict['spike_percentages']
        save_data(dirname, filename, net, all_parameters, all_train_loss,
                  all_train_accuracy, all_validate_loss, all_validate_accuracy,
                  validate_labels, mean_validate_outputs_sorted, std_validate_outputs_sorted,
                  spike_percentages, epoch_dir=(True, e_end))

        # evaluate on test set
        return_input = False
        # run again on training set (for spiketime saving)
        loss, final_train_accuracy, final_train_outputs, final_train_labels, final_train_inputs = validation_step(
            net, criterion, loader_train, device, return_input=return_input)
        loss, test_accuracy, test_outputs, test_labels, test_inputs = validation_step(
            net, criterion, loader_test, device, return_input=return_input)

        save_result_spikes(dirname, filename, final_train_outputs, final_train_labels, final_train_inputs,
                           test_outputs, test_labels, test_inputs, epoch_dir=(True, e_end))

        # each savepoint needs config to be able to run inference for eval
        save_config(dirname, filename, neuron_params, network_layout, training_params, epoch_dir=(True, e_end))
        numpy_rand_state = np.random.get_state()
        torch_rand_state = torch.get_rng_state()
        save_optim_state(dirname, filename, optimizer, scheduler, numpy_rand_state,
                         torch_rand_state, epoch_dir=(True, e_end))
    print("Training finished")

    print('####################')
    print('Test accuracy: {}'.format(test_accuracy))

    save_data(dirname, filename, net, all_parameters, all_train_loss,
              all_train_accuracy, all_validate_loss, all_validate_accuracy,
              validate_labels, mean_validate_outputs_sorted, std_validate_outputs_sorted, spike_percentages)

    return net
