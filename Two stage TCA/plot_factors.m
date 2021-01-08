% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file is part of the code for "Multimodal neural recordings with Neuro-FITM uncover
% diverse patterns of cortical-hippocampal interactions" published in Nature Neuroscience.
% You may use, change, or redistribute this code for non-commercial purposes.
% (C) Xin Liu, Kuzum Lab, University of California San Diego
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code plots the TCA results (region, time, and trial factors)
% Inputs:
%   - region_factor: region_factor
%   - time_factor: time_factor
%   - trial_factor: trial_factor
%   - Allen_ROIs: name of the cortical regions
%   - R_dim: the number of components
% Output:
%   - h: figure handle
function h = plot_factors(region_factor,time_factor,trial_factor,Allen_ROIs,R_dim,varargin)
params = inputParser;
params.addParameter('fig', []);
params.parse(varargin{:});
fig = params.Results.fig;

if isempty(fig)
    h = figure;
else
    h = fig;
end
axes = [];
ylims_region = [0,max(region_factor(:))];
ylims_time = [min(time_factor(:)),max(time_factor(:))];
ylims_trial = [min(trial_factor(:)),max(trial_factor(:))];
for i = 1:R_dim
    subaxis(R_dim,3,(i-1)*3+1,'SH',0.01,'SV',0.01,'MR',0.03,'ML',0.03,'MT',0.06,'MB',0.1);
    bar(region_factor(:,i)); hold on; xlim([0.5,16.5]); ylim(ylims_region);
%     plot(region_factor(:,i),'LineWidth',1.5); hold on; xlim([0.5,16.5]); ylim(ylims_region);
    axes(1) = gca;    
    subaxis(R_dim,3,(i-1)*3+2,'SH',0.01,'SV',0.01,'MR',0.03,'ML',0.03,'MT',0.06,'MB',0.1);
    plot(time_factor(:,i),'LineWidth',1.5); hold on; ylim(ylims_time);
    plot([0,0],get(gca,'ylim'),'--k','LineWidth',1.5); 
    axes(2) = gca;
    subaxis(R_dim,3,(i-1)*3+3,'SH',0.01,'SV',0.01,'MR',0.03,'ML',0.03,'MT',0.06,'MB',0.1);
    plot(trial_factor(:,i),'.'); hold on; ylim(ylims_trial);
    axes(3) = gca;
  
    if i == 1
        title(axes(1),'Region factor','FontSize',10);
        title(axes(2),'Time factor','FontSize',10);
        title(axes(3),'Trial factor','FontSize',10);
    end
    if i <= R_dim
        set(axes(1),'XTick',[],'YTick',[]);
        set(axes(2),'XTick',[],'YTick',[]);
        set(axes(3),'XTick',[],'YTick',[]);
    end
    if i == R_dim
        set(axes(1),'XTick',1:16,'XTickLabel',Allen_ROIs,'xticklabelrotation',45);
        set(axes(1),'FontSize',10);
        xlabel(axes(2),'Time (s)','FontSize',10,'FontWeight','bold');
        xlabel(axes(3),'Trials','FontSize',10,'FontWeight','bold');
    end
end
end