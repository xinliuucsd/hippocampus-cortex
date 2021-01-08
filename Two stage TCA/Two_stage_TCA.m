% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file is part of the code for "Multimodal neural recordings with Neuro-FITM uncover
% diverse patterns of cortical-hippocampal interactions" published in Nature Neuroscience.
% You may use, change, or redistribute this code for non-commercial purposes.
% (C) Xin Liu, Kuzum Lab, University of California San Diego
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code implements the two-stage TCA algorithm. This code uses estabilished community detection code
% (http://netwiki.amath.unc.edu/GenLouvain/GenLouvain).

%% Perform stage I TCA
reptime = 10;
% load the fluorescence data tensor
load('peri_ripple_dFF_segments.mat');
nregion = size(dFF_peri_ripple,1);
T = size(dFF_peri_ripple,2);
% dimension r x t x n, r: cortical regions, t: time steps, n: ripple trials
Ca_segs_all = dFF_peri_ripple - min(dFF_peri_ripple(:)) + 0.001; % rescale the data and add some small value to make it positive
error_all = zeros(1,reptime);
TCA_result_all = cell(1,reptime);
R_dim = 15; % specify the rank of the TCA model
parfor r = 1:reptime % run code in parallel to speed up the calculation
    tic;
    rndseed = r;
    TCA_result_all{r} = perform_TCA(Ca_segs_all, R_dim,'rayleigh',rndseed);
    error_all(r) = TCA_result_all{r}.error;
    fprintf('Finished %3.d/%3.d %3.1f\n',r,reptime,toc);
end
% find the best-fit model
[~,bestfitID] = min(error_all);
best_fit_all = bestfitID;
ind_left = setdiff(1:reptime,bestfitID);

% Perform the community detection to find consistent patterns
patterns = {};
region_factors = {};
time_factors = {};
trial_factors = {};

patterns = zeros(nregion,T,R_dim*reptime);
region_factors = zeros(nregion,R_dim,reptime);
time_factors = zeros(T,R_dim,reptime);
trial_factors = zeros(size(TCA_result_all{1,1}.trial_factor,1),R_dim,reptime);
for i = 1:reptime
    inds = ((i-1)*R_dim+1):(i*R_dim);
    patterns(:,:,inds) = TCA_result_all{i}.templates;
    region_factors(:,:,i) = TCA_result_all{i}.region_factor;
    time_factors(:,:,i) = TCA_result_all{i}.time_factor;
    trial_factors(:,:,i) = TCA_result_all{i}.trial_factor;
end

% Compute the adjacency matrix
adMat = {};
pat_num = R_dim*reptime;
adMat = zeros(pat_num,pat_num);
for i = 1:pat_num
    for j = 1:pat_num
        adMat(i,j) = corr2(patterns(:,:,i),patterns(:,:,j));
    end
end
% threshold the adjacency matrix
thresh = 0.7;
adMat = (adMat>thresh) .* adMat;
% Visualize the 2D correlation matrix
figure; imagesc(adMat); colorbar; colormap(flipud(gray));
xlabel('Pattern ID'); ylabel('Pattern ID');set(gca,'FontSize',12,'FontWeight','bold');

% Compute the modularity matrix and then the clustering
rng(10);
gamma = 1;
[B,twom] = modularity(adMat,gamma);
[S,Q]= genlouvain(B);
Q=Q/twom;
clust_ID = unique(S);
clust_size = zeros(1,length(clust_ID));
for i = 1:length(clust_ID)
    clust_size(i) = sum(S == i);
end
% plot the number of clustered patterns
figure; bar(clust_size); xlabel('Pattern cluster ID'); ylabel('Count');
set(gca,'FontSize',14,'FontWeight','bold');
% Examine the patterns that are clustered together
patterns_cluster = cell(1,length(clust_ID));
pat_recur_ID = find(clust_size >= 5); % only keep the clusters with a sufficiently large number of members
templates_clust = zeros(16,90,length(pat_recur_ID));
for i = 1:length(pat_recur_ID)
    pat_ID = find(S == clust_ID(pat_recur_ID(i)));
    templates_clust(:,:,i) = mean(patterns(:,:,pat_ID),3);
    patterns_cluster{i} = patterns(:,:,pat_ID);
end

% Plot the consistent clustered TCA patterns
% reorganize the templates and plot the template patterns
templates_clust_reorg = templates_clust;
templates_clust_reorg(1:8,:,:) = templates_clust(1:2:end,:,:);
templates_clust_reorg(9:16,:,:) = templates_clust(2:2:end,:,:);
fig = plot_TCA_templates(templates_clust_reorg,Allen_ROIs);

% Reorganize the data and compute correlation map
patterns_new = patterns;
adMat_new = adMat;
base = 0;
for i = 1:length(clust_ID)
    inds = base+ (1:sum(S == i));
    base = base + sum(S == i);
    patterns_new(:,:,inds) = patterns(:,:,S == i);
end
for i = 1:pat_num
    for j = 1:pat_num
        adMat_new(i,j) = corr2(patterns_new(:,:,i),patterns_new(:,:,j));
    end
end
% threshold the adjacency matrix
adMat_new = (adMat_new > thresh) .* adMat_new;
figure; imagesc(adMat_new); colorbar; colormap(flipud(gray));
xlabel('Pattern ID'); ylabel('Pattern ID'); set(gca,'FontSize',14,'FontWeight','bold');

% plot the patterns that are clustered together
% for i =1:size(templates_clust,3)
%     figure;
%     for k = 1:5
%         subplot(1,5,k);
%         temp = squeeze(patterns_cluster{i}(:,:,k));
%         temp(1:8,:) = squeeze(patterns_cluster{i}(1:2:end,:,k));
%         temp(9:end,:) = squeeze(patterns_cluster{i}(2:2:end,:,k));
%         imagesc(temp);colormap jet;
%         set(gca,'Ydir','normal');
%     end
% end

%% Perform stage II TCA
nd = ndims(Ca_segs_all);
sz = size(Ca_segs_all);
nc = length(pat_recur_ID);  % the rank used in paper is 8 for the combined data from 6 animals

% perform 10 times TCA with partially fixed initialization M0
reptimes = 10;
error_all = zeros(1,reptimes);
var_explain = zeros(1,reptimes);
TCA_info = {};
templates = {};
error_all_refined = zeros(1,reptimes);
TCA_info_refined = {};
templates_refined = {};
var_explain_refined = zeros(1,reptimes);

X = tensor(Ca_segs_all);
baseLine = repmat(mean(Ca_segs_all,3),1,1,size(Ca_segs_all,3));
B = tensor(baseLine);

for r = 1:reptimes
    % perform refined TCA with order 8
    rng(r);
    rndseed = r;
    % Initialize the TCA patterns
    Uinit = cell(nd,1);
    Uinit{1} = normalize(squeeze(mean(templates_clust,2)),'range'); 
    Uinit{2} = normalize(squeeze(mean(templates_clust,1)),'range'); 
    Uinit{3} = rand(sz(3),nc);
    M0 = ktensor(Uinit);
    M0 = M0 * (norm(X)/norm(M0)); % normalize
    tic;
    TCA_info_refined{r} = perform_TCA(Ca_segs_all, nc, 'rayleigh', rndseed,'init',M0);
    templates_refined{r} = TCA_info_refined{r}.templates;
    error_all_refined(r) = TCA_info_refined{r}.error;
    var_explain_refined(r) = 1 - (norm(X-TCA_info_refined{r}.M)^2 / norm(X-B)^2);
    fprintf('Finished %3.d/%3.d time %3.1d\n',r,reptimes,toc);
end

%% Plot the TCA patterns with lowest error
[~,best_id] = min(error_all_refined);
region_factor = TCA_info_refined{best_id}.region_factor;
time_factor = TCA_info_refined{best_id}.time_factor;
trial_factor = TCA_info_refined{best_id}.trial_factor;
template = TCA_info_refined{best_id}.templates;
R_dim = size(trial_factor,2);
fig = plot_factors(region_factor,time_factor,trial_factor,Allen_ROIs,R_dim);
fig = plot_TCA_templates(template,Allen_ROIs);