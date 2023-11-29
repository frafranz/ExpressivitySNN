import matplotlib.pyplot as plt
import time
import torch
import sys
print(sys.path)
sys.path.append('/Users/fhaniel/Desktop/Studium/Berlin/Rest/AI/SNN/Code/expressivity/TrainingSNNLinearActivation/src')


from networks.relu_network import ReLUNet
from networks.spiking_network import SpikingNet
from spiking_layer import SpikingLayer

"""
Plot the required runtime for different hidden layer sizes, 
to compare with the estimated time complexity for the SNN simulation and the ReLU network.

The results are plotted.
"""

if __name__ == '__main__':
    print("Start runtime tests for evaluation")

    # setup parameters
    layer_sizes = [10, 20, 30, 40, 60, 80, 100, 150, 200, 250, 300, 400, 500, 1000, 1200]
    N = 100 # number of repeated evaluations

    delays_init = (-2.3, 0.5) # mean and std of log of delay
    weights_init = (0., 0.5) # mean and std of weight
    thresholds_init = (1.0, 0.0) # mean and std of threshold

    sim_params = {
        'activation': 'linear', 
        'substitute_delay': True
    }

    print("ReLU network")
    runtimes_relu = []
    std_relu = []
    for size in layer_sizes:
        print("\tsize: ", size)

        network_layout = {
            'n_inputs': size,
            'layer_sizes': [size],
            'bias': [True],
            'n_layers': 1,
            'initialization': 'He'
        }
        net = ReLUNet(network_layout=network_layout, sim_params=sim_params)

        times = torch.zeros(N)
        data = torch.rand((N,size))
        for i in range(N):
            start = time.time()
            net(data[i])
            stop = time.time()
            times[i] = stop-start
        runtimes_relu.append(times.mean())
        std_relu.append(times.std())


    print("Spiking network")
    runtimes_spiking = []
    std_spiking = []
    for size in layer_sizes:
        print("\tsize: ", size)
        layer = SpikingLayer(
            input_features=size, output_features=size, sim_params=sim_params, 
            weights_init=weights_init, delays_init=delays_init, thresholds_init=thresholds_init,
            device='cpu'
        )

        times = torch.zeros(N)
        data = torch.rand((N,size))
        for i in range(N):
            start = time.time()
            layer(data[i].unsqueeze(0))
            stop = time.time()
            times[i] = stop-start
        runtimes_spiking.append(times.mean())
        std_spiking.append(times.std())


    print("Plotting results")
    fig, ax = plt.subplots()
    ax.errorbar(layer_sizes, runtimes_relu, yerr=std_relu, label="ReLU")
    fig.savefig("eval_complexity_relu.png", dpi=300)
    ax.errorbar(layer_sizes, runtimes_spiking, yerr = std_spiking,label="spiking")
    fig.savefig("eval_complexity_snn.png", dpi=300)