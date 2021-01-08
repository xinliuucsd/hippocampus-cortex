% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file is part of the code for "Multimodal neural recordings with Neuro-FITM uncover
% diverse patterns of cortical-hippocampal interactions" published in Nature Neuroscience.
% You may use, change, or redistribute this code for non-commercial purposes.
% (C) Xin Liu, Kuzum Lab, University of California San Diego
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code implements the pairwise decoding of cortical pattern types using the neuron firing activity during ripples

%% Load the data
load('data.mat');
% neuron_id: index of each neuron
% rp_trial_id: index of each ripple event assgiend to eight clsuters
% raster_data: the firing activity for each neuron during ripple events

%% Identify the important neurons by RFE algorithm
n_shuffle = 200;
kernel_size = 1;
rng(0);
% Now perform the decoding
neuron_importance = struct;
n_neuron = length(raster_data);
pattern_pairs = nchoosek(1:8,2);
neuron_importance.neuron_order_pat = zeros(n_neuron,size(pattern_pairs,1));
neuron_importance.CV_ba = zeros(size(pattern_pairs,1),n_neuron);
neuron_importance.subset_best = cell(1,size(pattern_pairs,1));
neuron_importance.CV_ba_best = zeros(size(pattern_pairs,1),1);
neuron_importance.CV_cf = zeros(size(pattern_pairs,1),n_neuron,2,2);
neuron_importance.pairs_compare = [];
neuron_importance.pvals = nan(28,28);
for pair = 1:size(pattern_pairs,1)
    tic;
    % compute the neuron spike features and the cortical pattern type for decoding
    n_trial_pair = zeros(1,2);
    ps = pattern_pairs(pair,:);
    n_trials = 0;
    for p = 1:size(ps,2)
        n_trials = n_trials + length(rp_trial_id{ps(p)});
    end
    neuron_importance.pairs_compare = [neuron_importance.pairs_compare; ps];
    neuron_pattern = zeros(size(neuron_id,2),n_trials);
    cort_type = zeros(1,n_trials);
    base = 0;
    for p = 1:size(ps,2)
        ind_temp = rp_trial_id{ps(p)};
        inds = base + (1:length(ind_temp));
        base = base + length(inds);
        for n = 1:size(neuron_id,2)
            neuron_pattern(n,inds) = 100*raster_data{neuron_id(n)}(ind_temp,11);
        end
        cort_type(inds) = p;
    end
    % obtain the cost matrix
    ratio_temp = sum(cort_type == 1) / length(cort_type);
    costmat = [0,1-ratio_temp;ratio_temp,0];
    % fit the whole model and obtain the weights
    [feat_list,CV_ba,CV_cf,ws,X_data,Y_data] = SVM_REF(neuron_pattern',cort_type',costmat,kernel_size,1);
    neuron_importance.neuron_order_pat(:,pair) = feat_list';
    neuron_importance.CV_cf(pair,:,:,:) = CV_cf;
    neuron_importance.CV_ba(pair,:) = CV_ba;
    [a,b] = max(CV_ba);
    neuron_importance.weight_best{pair} = ws{b};
    neuron_importance.X_data{pair} = X_data{b};
    neuron_importance.Y_data{pair} = Y_data{b};
    neuron_importance.CV_ba_best(pair) = a;
    neuron_importance.subset_best{pair} = feat_list(1:b);

    fprintf('Finished %3.d/%3.d %3.1fs\n',pair,28,toc);
end

%% Identify the significantly decodable pairs with the best subset of neurons
pairs_all = neuron_importance.pairs_compare;
neuron_importance.CV_ba_null = zeros(28,n_shuffle);
current_pair = 0;
for pair = 1:size(pattern_pairs,1)
    tic;
    % compute the neuron spike features and the cortical pattern type for decoding
    current_pair = current_pair + 1;
    ps = pairs_all(current_pair,:);
    feat_id = neuron_importance.subset_best{pair};
    n_trials = 0;
    for p = 1:size(ps,2)
        n_trials = n_trials + length(rp_trial_id{ps(p)});
    end
    neuron_pattern = zeros(size(neuron_id,2),n_trials);
    cort_type = zeros(1,n_trials);
    base = 0;
    for p = 1:size(ps,2)
        ind_temp = rp_trial_id{ps(p)};
        inds = base + (1:length(ind_temp));
        base = base + length(inds);
        for n = 1:size(neuron_id,2)
            neuron_pattern(n,inds) = 100*raster_data{neuron_id(n)}(ind_temp,11);
        end
        cort_type(inds) = p;
    end
    ratio_temp = sum(cort_type == 1) / length(cort_type);
    costmat = [0,1-ratio_temp;ratio_temp,0];
    cort_types = zeros(n_shuffle, length(cort_type));
    rng(666);
    for s = 1:n_shuffle
        cort_types(s,:) = cort_type(randperm(length(cort_type)));
    end
    confMats_temp = zeros(n_shuffle,2,2);
    neuron_pat_new = neuron_pattern(feat_id,:);
    parfor s = 1:n_shuffle
        % fit the whole model and obtain the weights
        Model = fitcsvm(neuron_pat_new',cort_types(s,:)','Standardize',true,'KernelFunction','linear',...
            'KernelScale', kernel_size,'Cost',costmat,'BoxConstraint',1);
        % construct the model
        rng(pair*n_shuffle+s+200000);
        CVSVMModel = crossval(Model,'leaveout','on');  % 10-fold cross validation
        % CVSVMModel = crossval(Model,'kfold',5);
        [pred_type,scorePred,cost] = kfoldPredict(CVSVMModel);
        % Now compute the performance
        [confMat,~] = confusionmat(cort_types(s,:)',pred_type);
        confMats_temp(s,:,:) = confMat;
    end
    for s = 1:n_shuffle
        confMat = squeeze(confMats_temp(s,:,:));
        TPR = confMat(1,1) / sum(confMat(:,1));
        TNR = confMat(2,2) / sum(confMat(:,2));
        neuron_importance.CV_ba_null(pair,s) = (TPR + TNR) / 2;
    end
    fprintf('Finished %3.d/%3.d %3.1fs\n',current_pair,size(pairs_all,1),toc);
end

%% Plot the significantly decoded pattern pairs within each mouse
figure;
neuron_importance.signi_pair = [];
neuron_importance.insigni_pair = [];
b_accuracy_plot = nan(8,8);
for pair = 1:size(pattern_pairs,1)
    ps = pattern_pairs(pair,:);
    b_accuracy_plot(ps(1),ps(2)) = neuron_importance.CV_ba_best(pair);
end
b_accuracy_plot(1:8+1:end) = nan;
B = [[b_accuracy_plot,zeros(8,1)];zeros(1,9);];
N = 64;
Color1 = [[linspace(0,1,N/2)',linspace(0,1,N/2)',linspace(0,1,N/2)'];...
    [ones(N/2,1),linspace(1,0,N/2)',linspace(1,0,N/2)']];
pattern_names = {};
for p = 1:8
    pattern_names{p} = sprintf('P%1d',p);
end
h = pcolor(B); set(h, 'EdgeColor', 'none');
cbar = colorbar; title(cbar,'b.acc.'); colormap(Color1); caxis([0,1]);
set(gca,'xaxisLocation','top');
set(gca, 'ydir', 'reverse');
set(gca,'XTick',1.5:1:8.5,'XTickLabel',pattern_names);
set(gca,'YTick',1.5:1:8.5,'YTickLabel',pattern_names);
axis square;
locs = 1.5:1:8.5;
for pair = 1:size(pattern_pairs,1)
    if isnan(neuron_importance.CV_ba_best(pair))
        [];
    else
        ps = pattern_pairs(pair,:);
        null_accuracy = neuron_importance.CV_ba_null(pair,:);
        % compute the p value
        accuracy = neuron_importance.CV_ba_best(pair);
        pvals = sum(null_accuracy > accuracy) / length(null_accuracy);
        neuron_importance.pvals(ps(1), ps(2)) = pvals;
        if pvals < 0.001
            text(-0.3+locs(ps(2)),locs(ps(1)),'***','FontSize',10);
        elseif pvals < 0.01
            text(-0.3+locs(ps(2)),locs(ps(1)),'**','FontSize',10);
        elseif pvals < 0.05
            text(-0.3+locs(ps(2)),locs(ps(1)),'*','FontSize',10);
        end
        if pvals < 0.05
            neuron_importance.signi_pair = [neuron_importance.signi_pair; [ps(1), ps(2)]];
        else
            neuron_importance.insigni_pair = [neuron_importance.insigni_pair; [ps(1), ps(2)]];
        end
    end
end

