# Introduction
This is the set of code used for the multimodal analysis of neural signals recorded with Neuro-FITM shown in the paper below.
# Publication: 
Xin Liu, Chi Ren, Yichen Lu, Yixiu Liu, Jeong-Hoon Kim, Stefan Leutgeb, Takaki Komiyama, and Duygu Kuzum. *Multimodal neural recordings with Neuro-FITM uncover diverse patterns of cortical-hippocampal interactions*. Nature Neuroscience (2021).

# Usage guide
## Requirements
The code is written and tested in MATLAB 2019b. Running the code will require tensor toolbox (https://www.tensortoolbox.org/) and community detection toolbox (http://netwiki.amath.unc.edu/GenLouvain/GenLouvain).

## Repository folders
- Ripple detection: the threshold based ripple detection code and the sample data
- Two stage TCA: the two-stage TCA algorithm, the functions to plot the results, and the sample data
- Decoding of cortical pattern: the code for recursive feature elimination algorithm with SVM using the sample data (neuron firing and pattern types)

# Licence
Copyright (c) 2021 Xin Liu, Kuzum Lab, University of California San Diego
Licenced under the [MIT License](LICENSE).
