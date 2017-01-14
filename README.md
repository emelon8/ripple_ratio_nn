# ripple_ratio_nn
This artificial neural network predicts the normalized ripple ratio, which is defined as the normalized ratio of ripple power to the low frequency power, for a network of neurons. The ripple power is the integral of power of the frequency at the peak of the power spectrum +-20 Hz. Low frequency power is the integral of power in the range [4,50] Hz.

The network of neurons from which the power spectrum was taken from Brunel and Wang (2003) and includes 4000 pyramidal neurons (p) and 1000 interneurons (i), biophysically modeled with the LIF model. Their code was modified to include DC current input to each neuron type.

This artificial neural network takes the amount of DC current as an input, and predicts the normalized ripple ratio as the output.
