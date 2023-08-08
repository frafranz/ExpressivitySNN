import unittest
import time

from src import utils, training, datasets


class TestExperiment(unittest.TestCase):
    def test_experiment_train(self):
        """
        Test that the the training mode of the experiment runs without errors on the XOR dataset.
        """

        # set up datasets, configs
        dataset = "xor"
        neuron_params = {
            "activation": "linear",
            "g_leak": 1.0,
            "leak": 0.0,
            "tau_syn": 1.0,
            "threshold": 1.0,
        } 
        network_layout = {
            "bias_times": [0.9],
            "layer_sizes": [10, 2],
            "n_biases": [1, 1],
            "n_inputs": 4,
            "n_layers": 2,
            "weight_means": [1.5, 0.5],
            "weight_stdevs": [0.8, 0.8]
        }
        training_params = {
            "alpha": 0.005,
            "batch_size": 150,
            "batch_size_eval": 200,
            "beta": 1.,
            "enforce_cpu": False,
            "epoch_number": 15,
            "epoch_snapshots": [5, 10],
            "learning_rate": 0.005,
            "lr_scheduler": {"gamma": 0.95, "step_size": 20, "type": "StepLR"},
            "max_dw_norm": 0.2,
            "max_num_missing_spikes": [0.3, 0.0],
            "momentum": 0,
            "numpy_seed": 12345,
            "optimizer": "adam",
            "print_step_percent": 5.0,
            "resolution": 0.01,
            "sim_time": 4.0,
            "torch_seed": 2000,
            "training_noise": {"mean": 0.0, "std_dev": 0.2},
            "use_forward_integrator": False,
            "use_hicannx": False,
            "weight_bumping_exp": True,
            "weight_bumping_targeted": True,
            "weight_bumping_value": 0.0005,
            "xi": 0.2
        }
        dataset_train = datasets.XOR()
        dataset_val = datasets.XOR()
        dataset_test = datasets.XOR()

        # main code
        t_start = time.perf_counter()
        dirname = None
        filename = None
        net = training.train(training_params, network_layout, neuron_params,
                                dataset_train, dataset_val, dataset_test, dirname, filename)
        t_end = time.perf_counter()
        duration = t_end - t_start
        print('Training {0} epochs -> duration: {1} seconds'.format(training_params['epoch_number'], duration))



if __name__ == '__main__':
    unittest.main()




