% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file is part of the code for "Multimodal neural recordings with Neuro-FITM uncover
% diverse patterns of cortical-hippocampal interactions" published in Nature Neuroscience.
% You may use, change, or redistribute this code for non-commercial purposes.
% (C) Xin Liu, Kuzum Lab, University of California San Diego
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code performs the TCA based on the Tensor toolbox (https://www.tensortoolbox.org/).
% Inputs:
%   - Ca_segs: the dF/F activity tensor (region x time x ripple trials)
%   - R_dim: the number of components
%   - mod: 'rayleigh'. see tensor toolbox for more details
%   - rndseed: random seed
% Output:
%   - TCA_info: the TCA decomposition result
function TCA_info = perform_TCA(Ca_segs, R_dim, mod, rndseed, varargin)
params = inputParser;
params.addParameter('init', []);
params.parse(varargin{:});
M0_init = params.Results.init;

baseLine = repmat(mean(Ca_segs,3),1,1,size(Ca_segs,3));
X = tensor(Ca_segs);
B = tensor(baseLine);

rng(rndseed);
if isempty(M0_init)
    M = gcp_opt(X,R_dim,'type',mod,'printitn',-1); % state = rndseed 
else
    M = gcp_opt(X,R_dim,'type',mod,'printitn',-1,'init',M0_init); % state = rndseed 
end
fprintf('Final fit: %e \n',1 - norm(X-full(M))/norm(X));
vizopts = {'PlotCommands',{'bar','line','scatter'},...
    'ModeTitles',{'Cortical Region','Time','Trials'},...
    'BottomSpace',0.10,'HorzSpace',0.04,'Normalize',0};
% figure(1); info1 = viz(M,'Figure',1,vizopts{:});
TCA_info.M = M;
error_model = norm(X-full(M))/norm(X);
error_base = norm(X-full(B))/norm(X);
% print(gcf,[outfolder,'\ripple_cortex_TCA_result'],'-dpng');

%% Plot some example modes
region_factor = double(M.u{1});
time_factor = double(M.u{2});
trial_factor = double(M.u{3});
R_factor = double(M.lambda);
templates = zeros(size(region_factor,1),size(time_factor,1),R_dim);
for i = 1:size(region_factor,2)
    templates(:,:,i) = R_factor(i) * region_factor(:,i) * time_factor(:,i)';
end
TCA_info.time_factor = time_factor;
TCA_info.region_factor = region_factor;
TCA_info.R_factor = R_factor;
TCA_info.trial_factor = trial_factor;
TCA_info.templates = templates;
TCA_info.error = error_model;
TCA_info.error_base = error_base;
end