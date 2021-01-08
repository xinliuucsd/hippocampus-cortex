% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file is part of the code for "Multimodal neural recordings with Neuro-FITM uncover
% diverse patterns of cortical-hippocampal interactions" published in Nature Neuroscience.
% You may use, change, or redistribute this code for non-commercial purposes.
% (C) Xin Liu, Kuzum Lab, University of California San Diego
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code computes the morlet wavelet transformation to obtain the spectrogram of the input signal.
% Inputs:
%   - x: LFP data
%   - srate: sampling rate
%   - flo: lower limit of the frequency range
%   - fhi: higher limit of the frequency range
%   - deltaf: frequency step
%   - range_cycles: smooth parameter to control width of the Gaussian window
% Output:
%   - tf: the spectrogram
function tf = my_morlet(x,srate,flo,fhi,deltaf, range_cycles)

min_freq = flo;
max_freq = fhi;
num_frex = (fhi-flo+1)/deltaf;
frex = linspace(min_freq,max_freq,num_frex);

% other wavelet parameters
s = logspace(log10(range_cycles(1)),log10(range_cycles(end)),num_frex) ./ (2*pi*frex);

N_orig=length(x);
% now loop over trials...
N=2^(nextpow2(length(x)));
x=[x,zeros(1,N-length(x))];
% initialize output time-frequency data
tf = zeros(length(frex),N);
dataX = fft(x);
freq_samples=srate*1/N*(-N/2:N/2-1);

% loop over frequencies
for fi=1:length(frex)
    % create wavelet and get its FFT
    waveletX = sqrt(pi/(2*s(fi)^2))*exp(-2*pi^2*s(fi)^2*(freq_samples-frex(fi)*ones(1,N)).^2);
    waveletX = waveletX ./ max(waveletX);
    % run convolution
    as = ifft(ifftshift(waveletX) .* dataX);
    % compute the amplitude instead of power
    tf(fi,:) = 2*abs(as);
end

tf=tf(:,1:N_orig);
