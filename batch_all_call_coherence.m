function [mdlResults, params, inputs] = batch_all_call_coherence(vdCall,cell_ks,varargin)

pnames = {'outDir', 'baseDir', 'dataDir','fixed_ridge_ks','chunkSize'};
dflts  = {[],[],[],nan(length(cell_ks),1),NaN};
[outDir,baseDir,dataDir,fixed_ridge_ks,chunkSize] = internal.stats.parseArgs(pnames,dflts,varargin{:});

nCells = length(cell_ks);
inputs = {'call_on','call_ps_pca','bioacoustics'};
mdlParams = struct('onlyCoherence',false,'onlyFeats',false,'onlyStandardize',false,'onlyNeuralData',false,'permuteInput',[]);
% permute_idxs = [0 0 0; 1 0 0; 0 1 0; 0 0 1; 0 1 1; 1 0 1; 1 1 0; 1 1 1];
permute_idxs = zeros(1,3);
nPermutes = size(permute_idxs,1);

t = tic;
mdlResults = cell(nPermutes,nCells);
params = cell(nPermutes,nCells);
batParams = cell(1,nCells);

for k = 1:nCells
    cell_k = cell_ks(k);
    b = strcmp(vdCall.batNums,vdCall.batNum{cell_k});
    if isempty(baseDir)
        baseDir = vdCall.baseDirs{b};
    end
    batParams{k} = struct('batNum',vdCall.batNum{cell_k},'cellInfo',vdCall.cellInfo{cell_k},...
        'baseDir',baseDir,'expDate',vdCall.expDay(cell_k),'lfp_tt',[],...
        'lfp_channel_switch',[],'dataDir',dataDir,'expType',vdCall.expType,...
        'callType',vdCall.callType,'boxNum',vdCall.boxNums{b},'band_ridge_k',fixed_ridge_ks(k,:));
end

if isnan(chunkSize)
    nChunk = 1;
    cell_chunks = 1:nCells;
else
    nChunk = ceil(nCells/chunkSize);
    cell_chunks = 1:(nCells + chunkSize - rem(nCells,chunkSize));
    cell_chunks = reshape(cell_chunks,chunkSize,[])';
    cell_chunks(~ismember(cell_chunks,1:nCells)) = NaN;
end

for perm_k = 1:nPermutes
    mdlParams.permuteInput = permute_idxs(perm_k,:);
    for chunk_k = 1:nChunk
        chunk_cell_ks = cell_chunks(chunk_k,:);
        chunk_cell_ks = chunk_cell_ks(~isnan(chunk_cell_ks));
        for k = chunk_cell_ks
            try
                [mdlResults{perm_k,k}, params{perm_k,k}] = all_call_coherence(batParams{k},inputs,'ridge',mdlParams);
            catch err
                disp(err)
                [mdlResults{perm_k,k}, params{perm_k,k}] = deal(NaN);
            end
        end
        progress = 100*(perm_k/nPermutes);
        fprintf('%d %% of cells processed\n',round(progress));
        toc(t);
        
        if ~isempty(outDir)
            try
                lmResults = struct('mdlResults',{mdlResults},'params',{params},'inputs',{inputs},'permute_idxs',permute_idxs,'cell_ks',cell_ks);
                save([outDir 'lmResults_' datestr(date,'mmddyyyy') '.mat'],'-v7.3','-struct','lmResults');
            catch err
                keyboard
            end
        end
    end
end
end