import copy
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import unittest

import utils
from spiking_layer import SpikingLayer


class TestEventbasedVsDiscretized(unittest.TestCase):
    @classmethod
    def setUp(self) -> None:
        self.num_input = 120
        self.num_output = 30
        self.batch_size = 50
        self.input_binsize = 0.01  # doubles as initial resolution
        self.sim_time = 3.
        self.debug = False

        self.resols = self.input_binsize / 2**np.arange(0, 6)

        self.seed = np.random.randint(0, 100000)

        self.weight_mean = 0.06
        self.weights_std = self.weight_mean * 4.
        self.weights_normal = True

        self.sim_params_eventbased = {
            'activation': 'linear',
            'leak': 0.,
            'g_leak': 1.,
            'threshold': 1.,
            'tau_syn': 1.,
            # or e.g. the following, but adapt weights (self.weight_mean = 0.10)
            # 'leak': 0.7,
            # 'g_leak': 4.,
            # 'threshold': 1.08,
            # 'tau_syn': 1.,
        }

    # @unittest.skip("faster testing")
    def test_nograd(self):
        """
        Testing the forward pass of the network

        Using the fwd pass from utils.py with the analytical formula for a comparison
        with a numerical integrator.
        """
        print(f"####### Using self.seed {self.seed} ########")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = utils.get_default_device()

        sim_params_integrator = copy.deepcopy(self.sim_params_eventbased)
        sim_params_integrator.update({
            'use_forward_integrator': True,
            'resolution': self.input_binsize,
            'sim_time': self.sim_time
        })

        print("### generating data and weights")
        times_input = torch.empty((self.batch_size, self.num_input), device=device)
        # torch.nn.init.normal_(times_input, mean=0.2)
        torch.nn.init.uniform_(times_input, a=0.1, b=1.5)
        times_input = torch.abs(times_input)
        print("#### binning inputs")
        times_input = (times_input / self.input_binsize).round() * self.input_binsize
        if self.debug:
            print(f"using inputs {times_input}")

        weights = torch.empty((self.num_input, self.num_output), device=device)
        if self.weights_normal:
            torch.nn.init.normal_(weights, mean=self.weight_mean, std=self.weights_std)
        else:
            torch.nn.init.uniform_(weights, a=self.weight_mean - 2 * self.weights_std, b=self.weight_mean + 2 * self.weights_std)

        delays = torch.zeros_like(weights)
        thresholds = torch.ones(self.num_output)

        print("### generating layers")
        layer_eventbased = SpikingLayer(
            self.num_input, self.num_output, self.sim_params_eventbased, weights, delays, thresholds,
            device, 0)
        layer_integrator = SpikingLayer(
            self.num_input, self.num_output, sim_params_integrator, weights, delays, thresholds,
            device, 0)

        # eventbased will not change
        print("### one time eventbased forward pass")
        with torch.no_grad():
            outputs_eventbased = layer_eventbased(times_input)
            outputs_eventbased_inf = torch.isinf(outputs_eventbased)

        print("### looping integrator passed with different resolutions")
        differences_l1, differences_l2 = [], []
        int_infs = []
        for resol in self.resols:
            layer_integrator.sim_params['resolution'] = resol
            layer_integrator.sim_params['steps'] = int(np.ceil(sim_params_integrator['sim_time'] / resol))
            layer_integrator.sim_params['decay_syn'] = float(np.exp(-resol / sim_params_integrator['tau_syn']))
            layer_integrator.sim_params['decay_mem'] = float(np.exp(-resol / sim_params_integrator['tau_syn']))

            assert self.input_binsize >= layer_integrator.sim_params['resolution'], \
                "inputs are binned too weakly compared to resolution"

            print(f"#### forward pass for resol {resol}")
            with torch.no_grad():
                outputs_integrator = layer_integrator(times_input)

            # handle infs
            if self.debug:
                print(f"##### event has {outputs_eventbased_inf.sum()} infs, "
                      f"integrator has {torch.isinf(outputs_integrator).sum()} infs")
            int_infs.append(torch.isinf(outputs_integrator).sum())

            # mean_difference = torch.mean(torch.abs(outputs_eventbased - outputs_integrator))
            # max_difference = torch.max(torch.abs(outputs_eventbased - outputs_integrator))
            difference_nonnan = outputs_eventbased - outputs_integrator
            difference_nonnan[torch.isnan(difference_nonnan)] = 0
            difference_l1 = torch.sum(torch.abs(difference_nonnan))
            differences_l1.append(difference_l1)
            difference_l2 = torch.sqrt(torch.sum((difference_nonnan)**2))
            differences_l2.append(difference_l2)
            if self.debug:
                print(f"#####               difference_l1 {difference_l1}")
                print(f"#####               difference_l2 {difference_l2}")

        print("### plotting")
        differences_l1 = np.array(differences_l1) / (self.batch_size * self.num_output)
        differences_l2 = np.array(differences_l2) / (self.batch_size * self.num_output)
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.resols, differences_l1, marker='x', label="l1")
        ax.plot(self.resols, differences_l2, marker='x', label="l2")
        # ax.axvline(input_binsize, label="input_binsize")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("resolution of integrator")
        ax.set_title(f"bin size of input {self.input_binsize}, self.seed {self.seed}, {self.num_input} inputs\nweights"
                     f" {self.weight_mean}+-{self.weights_std}" if self.weights_normal else
                     f"in [{self.weight_mean - 2 * self.weights_std}, {self.weight_mean + 2 * self.weights_std}")
        ax.set_ylabel("normed difference per output time\n(/ batch_size / num_output)")
        ax2 = ax.twinx()
        int_infs = np.array(int_infs) / (self.batch_size * self.num_output)
        ax2.bar(self.resols, int_infs, alpha=0.4, width=0.5 * self.resols,
                label="number of infs in integrator", color='C3')
        ax2.axhline(float(outputs_eventbased_inf.sum()) / (self.batch_size * self.num_output),
                    label="number of infs in evba", color='C3')
        ax2.set_ylim(min(int_infs) - 0.1, max(int_infs) + 0.1)
        ax2.set_ylabel('fraction infs')

        ax.legend()
        fig.tight_layout()
        fig.savefig('debug_integrator.png')

        differences_l1[np.isinf(differences_l1)] = 1000.
        differences_l2[np.isinf(differences_l2)] = 1000.
        self.assertTrue(
            np.all(np.diff(differences_l1) <= 0),
            "The l1 norms are not decreasing with increasing integrator resolution, see plot.")
        self.assertTrue(
            np.all(np.diff(differences_l2) <= 0),
            "The l2 norms are not decreasing with increasing integrator resolution, see plot.")
        
    # @unittest.skip("faster testing")
    def test_nograd_with_delta(self):
        """
        Also test the integrator with non-infinite delta.

        In this case, there is no eventbased calculation available to compare with.
        """
        print(f"####### Using self.seed {self.seed} ########")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = utils.get_default_device()

        sim_params_integrator = copy.deepcopy(self.sim_params_eventbased)
        sim_params_integrator.update({
            'use_forward_integrator': True,
            'resolution': self.input_binsize,
            'sim_time': self.sim_time,
            'delta': 1
        })

        print("### generating data and weights")
        times_input = torch.empty((self.batch_size, self.num_input), device=device)
        # torch.nn.init.normal_(times_input, mean=0.2)
        torch.nn.init.uniform_(times_input, a=0.1, b=1.5)
        times_input = torch.abs(times_input)
        print("#### binning inputs")
        times_input = (times_input / self.input_binsize).round() * self.input_binsize
        if self.debug:
            print(f"using inputs {times_input}")

        weights = torch.empty((self.num_input, self.num_output), device=device)
        if self.weights_normal:
            torch.nn.init.normal_(weights, mean=self.weight_mean, std=self.weights_std)
        else:
            torch.nn.init.uniform_(weights, a=self.weight_mean - 2 * self.weights_std, b=self.weight_mean + 2 * self.weights_std)

        delays = torch.zeros_like(weights)
        thresholds = torch.ones(self.num_output)

        print("### generating integrator layer")
        layer_integrator = SpikingLayer(
            self.num_input, self.num_output, sim_params_integrator, weights, delays, thresholds,
            device, 0)

        print("### looping integrator with finite delta and with different resolutions")
        for resol in self.resols:
            layer_integrator.sim_params['resolution'] = resol
            layer_integrator.sim_params['steps'] = int(np.ceil(sim_params_integrator['sim_time'] / resol))
            layer_integrator.sim_params['decay_syn'] = float(np.exp(-resol / sim_params_integrator['tau_syn']))
            layer_integrator.sim_params['decay_mem'] = float(np.exp(-resol / sim_params_integrator['tau_syn']))

            assert self.input_binsize >= layer_integrator.sim_params['resolution'], \
                "inputs are binned too weakly compared to resolution"

            print(f"#### forward pass for resol {resol}")
            with torch.no_grad():
                outputs_integrator = layer_integrator(times_input)

            # handle infs
            if self.debug:
                print(f"##### integrator has {torch.isinf(outputs_integrator).sum()} infs")


if __name__ == '__main__':
    unittest.main()
