function [mdlResults, params, cell_ks, input] = batch_all_call_coherence(vdCall,cell_ks)
if nargin < 2
    cell_ks = [vdCall.responsive_cells_by_bat{[1 2 4]}];
end
nCells = length(cell_ks);
input = {'call_on'};
mdlParams = struct('onlyCoherence',false,'onlyFeats',false,'onlyStandardize',false,'onlyNeuralData',false,'permuteInput',[]);
permute_idxs = [0 0 0; 1 0 0; 0 1 0; 0 0 1; 0 1 1; 1 0 1; 1 1 0];
nPermutes = size(permute_idxs,1);
dataDir = 'C:\Users\phyllo\Documents\Maimon\ephys\data_analysis_results\data_for_export\all_call_spike_data\';

t = tic;
mdlResults = cell(nPermutes,nCells);
params = cell(nPermutes,nCells);
for perm_k = 1:nPermutes
    mdlParams.permuteInput = permute_idxs(perm_k,:);
    parfor k = 1:nCells
        cell_k = cell_ks(k);
        batParams = struct('batNum',vdCall.batNum{cell_k},'cellInfo',vdCall.cellInfo{cell_k},...
            'call_echo',vdCall.call_echo,'baseDir',vdCall.baseDirs{1},'expDate',vdCall.expDay(cell_k),...
            'lfp_channel_switch',[],'lfp_tt',[],'dataDir',dataDir,'expType','juvenile');
        try
            [mdlResults{perm_k,k}, params{perm_k,k}] = all_call_coherence(batParams,input,'ridge',mdlParams);
        catch err
            [mdlResults{perm_k,k}, params{perm_k,k}] = deal(NaN);
        end
        
        progress = 100*(k/nCells);
        fprintf('%d %% of cells processed\n',round(progress));
        toc(t);
        
        lastProgress = progress;
    end
end