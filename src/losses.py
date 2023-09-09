import numpy as np
import torch

from utils import to_device


class LossFunction(torch.nn.Module):
    """Custom loss function computing cross-entropy"""
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
    """Standard mean squared error loss function"""
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
        target = to_device(torch.eye(self.number_labels), self.device)[true_label.long()] * (self.t_correct - self.t_wrong) + self.t_wrong
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
    """Dynamically get the loss function depending on the params
    
    Parameters:
        training_params: parameters of the experiment, which contain the ones concerning loss
        number_labels: number of predictable classes
        tau_syn: synaptic decay time constant, the times for the correct and wrong class are scaled with this value
        device: the device to operate on

    Returns:
        the loss values computed with the selected function
    """
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