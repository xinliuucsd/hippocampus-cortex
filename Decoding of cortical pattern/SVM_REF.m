% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file is part of the code for "Multimodal neural recordings with Neuro-FITM uncover
% diverse patterns of cortical-hippocampal interactions" published in Nature Neuroscience.
% You may use, change, or redistribute this code for non-commercial purposes.
% (C) Xin Liu, Kuzum Lab, University of California San Diego
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code performs recursive feature elimination algorithm for SVM
% Inputs:
%   - X: input features (neuron firing counts)
%   - Y: the cortical pattern type
%   - c: cost matrix
%   - k: kernel scale
%   - bs: box constraint
% Output:
%   - feat_ordered: the order of neuron features
%   - CV_va: the cross-validated balanced accuracy
%   - ws: the weight for the best subset of neurons
%   - X_datas: the firing counts for best subset of neurons
%   - Y_data: same as Y
function [feat_ordered,CV_ba,CV_cf,ws,X_data,Y_data] = SVM_REF(X,Y,c,k,bs)

n_feature = size(X,2);
feat_ordered = [];
feat_left = 1:n_feature;

% Now start the recursive loop
while ~isempty(feat_left)
    % fit SVM
    Model = fitcsvm(X(:,feat_left),Y,'Standardize',true,'KernelFunction','linear',...
        'KernelScale', k,'Cost',c,'BoxConstraint',bs);
    weight = Model.Beta';
    crit = weight.^2;
    [~,b] = sort(crit,'ascend');
    % update the two feature lists
    feat_ordered = [feat_left(b(1)),feat_ordered];
    feat_left(b(1)) = [];
end

% Now examine the leave one out CV performance to determine the best set
CV_ba = zeros(1,n_feature);
CV_cf = zeros(n_feature,2,2);
ws = {};
X_data = cell(1,n_feature);
Y_data = cell(1,n_feature);
parfor n = 1:n_feature
    X_subset = X(:,feat_ordered(1:n));
    Model = fitcsvm(X_subset,Y,'Standardize',true,'KernelFunction','linear',...
        'KernelScale', k,'Cost',c);
    CVSVMModel = crossval(Model,'leaveout','on');  % leave-one-out cross validation
    [Yhat,~,~] = kfoldPredict(CVSVMModel);
    % Now compute the performance
    [cf,~] = confusionmat(Y,Yhat);
    CV_cf(n,:,:) = cf;
    TPR = cf(1,1) / sum(cf(:,1));
    TNR = cf(2,2) / sum(cf(:,2));
    CV_ba(n) = (TPR + TNR) / 2;
    ws{n} = Model.Beta;
    X_data{n} = X_subset;
    Y_data{n} = Y;
end