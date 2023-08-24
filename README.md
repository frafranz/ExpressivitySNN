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

For our default parameters, refer to `yin_yang.yaml` and `xor.yaml`. 

*To try out the training:*
- decide for one of the datasets (e.g. Yin-Yang or XOR)
- if needed, adapt the respective config file in the folder experiment_configs
- run the experiment.py script with parameters "train" and the config path

The command might look like this: 
`python experiment.py train ../experiment_configs/yin_yang.yaml`

The code also allows continuing the training of models or mere evaluation.

[1] Singh et al., 2023 \
[2] Goeltz et al., 2021
