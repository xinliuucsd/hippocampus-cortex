% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file is part of the code for "Multimodal neural recordings with Neuro-FITM uncover
% diverse patterns of cortical-hippocampal interactions" published in Nature Neuroscience.
% You may use, change, or redistribute this code for non-commercial purposes.
% (C) Xin Liu, Kuzum Lab, University of California San Diego
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code implements threshold based ripple detection algorithm on example LFP data.
% The result (ripple time, ripple duration/amplitude/frequency) is saved to mat file.

%% load the lfp data
out_lowpass_name = 'lfp_data_example.mat';
load(out_lowpass_name);

%% detect the ripple events and save to file
outputfile = 'ripple_result.mat';
fold = 2;

% bandpass filter the data at ripple range
d_band = designfilt('bandpassiir','FilterOrder',8, ...
         'HalfPowerFrequency1',100,'HalfPowerFrequency2',200, ...
         'SampleRate',fs_low);
% compute the envelope of the ripple band ECoG
ripple_ecog = filtfilt(d_band,data_low);
ripple_env_hilbert = abs(hilbert(ripple_ecog));
% compute the spectrogram of all the channels
ripple_spectro = my_morlet(data_low',fs_low, 1,210,1,[4,20])';
movement_pow = my_morlet(data_low',fs_low, 300,400,1,[4,20])'; % 300-400 Hz power
mean_move_pow = smoothdata(mean(movement_pow,2),'gaussian',fs_low);
move_thresh = 10; % Impirically, we find 300-400 Hz power increases are due to movement

% some initialization to store the ripple parameters
ripple_time = [];
ripple_dur = [];
ripple_freq = [];
ripple_amp = [];

% start finding ripples in each channel
threshold = fold*std(ripple_env_hilbert); 
base = mean(ripple_env_hilbert);
ripple_now = false;
% iterate through every data point in the ripple envelope
%for i = 1:length(ripple_env_hilbert)
for i = 1:length(ripple_env_hilbert)
    if (ripple_now == false && ripple_env_hilbert(i) > (base+threshold))
        j_start = i;
        % don't consider the initial 0.5 s segment
        if j_start < 0.5*fs_low
            continue;
        end
        % now loop backward to find the time where signal just cross
        % the baseline. Use that point as the ripple starting point
        while (ripple_env_hilbert(j_start)> base)
            j_start = j_start - 1;
            if j_start == 0 || j_start > length(ripple_env_hilbert)
                j_start = i;
                break;
            end
        end
        % attach the start time of the new ripple
        ripple_time = [ripple_time, j_start];
        ripple_now = true;
    end
    % detect the time when ripple pass below the base+threshold
    if ripple_now == true && ripple_env_hilbert(i) < (base+threshold)
        ripple_now = false;
        j_end = i;
        % loop forward to find the ripple end time when it goes below
        % the base value
        while (ripple_env_hilbert(j_end)> base)
            j_end = j_end + 1;
            if j_end == 0 || j_end > length(ripple_env_hilbert)
                j_end = length(ripple_env_hilbert);
                break;
            end
        end
        % find the peak ripple amplitude for this ripple event
        [temp_max,tmax] = max(ripple_env_hilbert(ripple_time(end):j_end));
        % select the peak amplitude to be greater than base+2*threshold
        if temp_max >= (base+2*threshold)
            % find the peak frequency through 20Hz sliding window
            fwindow1 = 0.02*fs_low;
            fwindow2 = 0.04*fs_low;
            frange1 = (90:210)';
            frange2 = (60:210)';
            pows = zeros(1,101);
            % compute the mean power at each frequency during the ripple time
            for f = 1:100
                pows(f) = mean(ripple_spectro(j_start:j_end,f+100));
            end
            % find the center frequency that has the largest magnitude
            [~,I] = max(pows);
            % compute the mean power in a narrow frequency range near the center frequency
            freq_ind1 = knnsearch(frange1,I+100,'K',fwindow1);
            % compute the mean power in a wide frequency range near the center frequency
            freq_ind2 = knnsearch(frange2,I+100,'K',fwindow2);
            center_portion = sum(sum(ripple_spectro(j_start:j_end,frange1(freq_ind1))));
            total = sum(sum(ripple_spectro(j_start:j_end,frange2(freq_ind2))));
            % we require that the ripple frequency power should have a
            % concentration of 60 percent
            if (center_portion/total >0.6) && mean_move_pow(j_end) < move_thresh % to remove movement artifacts
                ripple_freq = [ripple_freq, I+100];
                ripple_dur = [ripple_dur, (j_end-ripple_time(end))];
                ripple_amp = [ripple_amp, temp_max];
            else
                ripple_time(end) = [];
            end
        else
            ripple_time(end) = [];
        end
    end
end
% throw away the ripple events that have duration less than 20 ms
[~,ind_keep] = find(ripple_dur >= 0.02*fs_low);
ripple_keep_time = ripple_time(ind_keep);
ripple_keep_dur = ripple_dur(ind_keep);
ripple_keep_freq = ripple_freq(ind_keep);
ripple_keep_amp = ripple_amp(ind_keep);
% merge the ripple events that have same start time and duration
[~, ia1, ~] = unique(ripple_keep_time);
ripple_keep_time = ripple_keep_time(ia1);
ripple_keep_dur = ripple_keep_dur(ia1);
ripple_keep_freq = ripple_keep_freq(ia1);
ripple_keep_amp = ripple_keep_amp(ia1);
% record the LFP associated with ripple events 
ripple_LFP = {};
for i = 1:length(ripple_keep_time)
    interval = ripple_keep_time(i)+ (-round(0.05*fs_low):ripple_keep_dur(i)+round(0.05*fs_low));  % 50 ms before and after ripple event
    if max(interval) > length(data_low)
        ripple_keep_time(i) = [];
        ripple_keep_dur(i) = [];
        ripple_keep_freq(i) = [];
        ripple_keep_amp(i) = [];
        break;
    else
        ripple_LFP{i} = data_low(interval);
    end
end

%% Reorganize the result
ripple_channels.ripple_time = ripple_keep_time/fs_low; % in the unit of second
ripple_channels.ripple_dur = ripple_keep_dur/fs_low*1e3; % in the unit of milisecond
ripple_channels.ripple_freq = ripple_keep_freq; % in the unit of Hz
ripple_channels.ripple_amp = ripple_keep_amp; % in the unit of uV
ripple_channels.ripple_LFP = ripple_LFP; % in the unit of uV
%% examine the example ripple events
ripple_channels.ripple_onoff = [ripple_channels.ripple_time',ripple_channels.ripple_time'+ripple_channels.ripple_dur'];
save(outputfile,'ripple_channels','-v7.3');
