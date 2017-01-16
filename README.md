# ripple_ratio_nn
This artificial neural network predicts the normalized ripple ratio, which is defined as the normalized ratio of ripple power to the low frequency power, for a network of neurons. The ripple power is the integral of power of the frequency at the peak of the power spectrum ±20 Hz. Low frequency power is the integral of power in the range [4,50] Hz.

The network of neurons from which the power spectrum was taken from Brunel and Wang (2003) and includes 4000 pyramidal neurons (p) and 1000 interneurons (i), biophysically modeled with the LIF model. Their code was modified to include DC current input to each neuron type.

This artificial neural network takes the amount of DC current as an input, and predicts the normalized ripple ratio as the output.

The code in ripple_ratio_nn.py takes normalized ripple ratio and DC current data from data/rippleratio_data.npz and uses it for both training and testing. Ripple frequencies are also used from data/additional_inputs.npz for training and testing. The ripple ratio data is randomly divided up accoring to the proprotion that the user defines to be used for training and testing (train_pct). Users can also edit the number of different training set sizes to try (dataset_sizes), for the creation of learning curves, and the number of learning iterations.

The output is a single number, which is a prediction of the normalized ripple ratio. This is then compared against the real ripple ratio (given the DC inputs to both types of neurons and the ripple frequency), plus or minus some permissable error (correct_threshold).
