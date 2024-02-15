# ExpressivitySNN

An implementation of the spiking neural network model defined in "Expressivity of Spiking Neural Networks" [1].
We make use of **PyTorch** and vectorize the computations where possible.
The forward pass includes an explicit calculation of spiketimes, which is possible due to the simplicity of the model. 
For the backward pass, gradients are custom-defined since the spiketime computations are not classically differentiable. 
We adapted the code base created by the authors of "Fast and energy-efficient neuromorphic deep learning
with first-spike times" [2] to fit our model. Major additions to this codebase are: 
- trainable delay parameters between neurons (possibly log transformed to ensure the positivity of the delays)
- trainable threshold parameters
- tests covering the forward and backward pass for small networks
- a time discretized implementation of the network, allowing for different values of $\delta$ in our model
- an arbitrary number of layers rather than just two.

For different sets of default parameters, refer to `yin_yang.yaml`, `yin_yang_relu.yaml` and `yin_yang_3_layer.yaml`.

*To try out the training:*
- decide for one of the datasets (e.g. Yin-Yang or XOR)
- if needed, adapt the respective config file in the folder `experiment_configs`
- run the `experiment.py` script with parameters `train` and the path of the config.

The command might look like this: 
`python experiment.py train ../experiment_configs/yin_yang.yaml`

The test suite can be run from the folder `src`, via
`python -m unittest`.

The code also allows to continue the training of models or mere evaluation.

[1] Singh et al., 2023 \
[2] Goeltz et al., 2021
