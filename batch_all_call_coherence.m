function [mdlResults, params, cell_ks, input] = batch_all_call_coherence(vdCall,cell_ks,dataDir,output_folder)
if nargin < 2
    cell_ks = [vdCall.responsive_cells_by_bat{[1 2 4]}];
    dataDir = 'C:\Users\phyllo\Documents\Maimon\ephys\data_analysis_results\data_for_export\all_call_spike_data\';
    output_folder = [];
elseif nargin <3
    dataDir = 'C:\Users\phyllo\Documents\Maimon\ephys\data_analysis_results\data_for_export\all_call_spike_data\';
    output_folder = [];
elseif nargin<4
    output_folder = [];
end
nCells = length(cell_ks);
input = {'call_on','call_ps_pca','bioacoustics'};
mdlParams = struct('onlyCoherence',false,'onlyFeats',false,'onlyStandardize',false,'onlyNeuralData',false,'permuteInput',[]);
permute_idxs = [0 0 0; 1 0 0; 0 1 0; 0 0 1; 0 1 1; 1 0 1; 1 1 0; 1 1 1];
nPermutes = size(permute_idxs,1);

t = tic;
mdlResults = cell(nPermutes,nCells);
params = cell(nPermutes,nCells);
batParams = cell(1,nCells);

for k = 1:nCells
    cell_k = cell_ks(k);
    batParams{k} = struct('batNum',vdCall.batNum{cell_k},'cellInfo',vdCall.cellInfo{cell_k},...
        'call_echo',vdCall.call_echo,'baseDir',vdCall.baseDirs{1},'expDate',vdCall.expDay(cell_k),...
        'lfp_channel_switch',[],'lfp_tt',[],'dataDir',dataDir,'expType','juvenile');
end

for perm_k = 1:nPermutes
    mdlParams.permuteInput = permute_idxs(perm_k,:);
    parfor k = 1:nCells
        try
            [mdlResults{perm_k,k}, params{perm_k,k}] = all_call_coherence(batParams{k},input,'ridge',mdlParams);
        catch err
            disp(err)
            [mdlResults{perm_k,k}, params{perm_k,k}] = deal(NaN);
        end        
    end
    progress = 100*(perm_k/nPermutes);
    fprintf('%d %% of cells processed\n',round(progress));
    toc(t);
end

if ~isempty(output_folder)
   try
       lmResults = struct('mdlResults',{mdlResults},'params',{params},'inputs',{input},'permute_idxs',permute_idxs,'cell_ks',cell_ks);
       save([output_folder 'lmResults_' datestr(date,'mmddyyyy') '.mat'],'-v7.3','-struct','lmResults');
   catch err
       keyboard
   end
end

end