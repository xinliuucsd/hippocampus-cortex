% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file is part of the code for "Multimodal neural recordings with Neuro-FITM uncover
% diverse patterns of cortical-hippocampal interactions" published in Nature Neuroscience.
% You may use, change, or redistribute this code for non-commercial purposes.
% (C) Xin Liu, Kuzum Lab, University of California San Diego
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code plots the pattern templates obtained from region factor and time factor
% Inputs:
%   - template: the cortical pattern templates
%   - Allen_ROIs: name of the cortical regions
% Output:
%   - fig: figure handle
function fig = plot_TCA_templates(template,Allen_ROIs)

dur = size(template,2);
R_dim = size(template,3);
[XX,YY] = meshgrid(1:(dur+1),1:17);
template_reorg = template;
template_reorg(1:8,:,:) = template(1:2:end,:,:);
template_reorg(9:16,:,:) = template(2:2:end,:,:);

%% plot the template in 2D colormap
fig = figure;
for i = 1:R_dim
    subplot(1,R_dim,i);
    tempdata_before = zeros(size(squeeze(template_reorg(:,:,i))) + 1);
    tempdata_before(1:end-1,1:end-1) = squeeze(template_reorg(:,:,i));
    h=surf(XX,YY,tempdata_before);view(2);
    if i == 1
        set(gca,'ytick',1.5:17.5,'YTickLabel',{Allen_ROIs{1:2:end},Allen_ROIs{2:2:end}}); ylim([1,17]);
    else
        set(gca,'ytick',[]);  ylim([1,17]);
    end
    set(h, 'EdgeColor', 'none');
    colormap jet;set(gca,'FontSize',8,'FontWeight','bold');
    title(['Pattern ',num2str(i)]);
end

end