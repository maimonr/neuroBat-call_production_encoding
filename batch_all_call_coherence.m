function [mdlResults, params, inputs] = batch_all_call_coherence(vdCall,cell_ks,varargin)

pnames = {'output_folder', 'baseDir', 'fixed_ridge_ks','dataDir','chunkSize','output_fname','lmResults'}; 
dflts  = {pwd,[], nan(length(cell_ks),1),[],NaN,[],[]}; 
[output_folder,baseDir,fixed_ridge_ks,dataDir,chunkSize,output_fname,lmResults] = internal.stats.parseArgs(pnames,dflts,varargin{:}); 
 
if isempty(output_fname)
    output_fname = fullfile(output_folder,['lmResults_' datestr(date,'mmddyyyy') '.mat']);
end

if isempty(lmResults)
    inputs = {'call_on','call_ps_pca','bioacoustics'};
    mdlParams = struct('onlyCoherence',false,'onlyFeats',false,'onlyStandardize',false,'onlyNeuralData',false,'permuteInput',[]);
    permute_idxs = [0 0 0; 1 0 0; 0 1 0; 0 0 1; 0 1 1; 1 0 1; 1 1 0; 1 1 1];
    start_perm_k = 1;
    start_cell_k = 1;
    
    nCells = length(cell_ks);
    nPermutes = size(permute_idxs,1);
    
    mdlResults = cell(nPermutes,nCells);
    params = cell(nPermutes,nCells);
else
    assert(all(lmResults.cell_ks == cell_ks))
    permute_idxs = lmResults.permute_idxs;
    
    nCells = length(cell_ks);
    nPermutes = size(permute_idxs,1);
    
    mdlResults = lmResults.mdlResults;
    params = lmResults.params;
    inputs = lmResults.inputs;
    
    mdlParams = lmResults.params{1}.mdlParams;
    mdlParams.permuteInput = [];
    start_ind = find(cellfun(@isempty,lmResults.mdlResults)',1,'first');
    [start_cell_k,start_perm_k] = ind2sub(size(lmResults.mdlResults'),start_ind);   
end

t = tic;

batParams = cell(1,nCells);

for k = 1:nCells
    cell_k = cell_ks(k);
    b = strcmp(vdCall.batNums,vdCall.batNum{cell_k});
    if isempty(baseDir)
        baseDir = vdCall.baseDirs{b};
    end
    batParams{k} = struct('batNum',vdCall.batNum{cell_k},'cellInfo',vdCall.cellInfo{cell_k},...
        'baseDir',baseDir,'expDate',vdCall.expDay(cell_k),'lfp_tt',[],...
        'lfp_channel_switch',[],'dataDir',dataDir,'expType',vdCall.expType{b},...
        'callType',vdCall.callType,'boxNum',vdCall.boxNums{b},...
        'selectCalls',vdCall.selectCalls,'band_ridge_k',fixed_ridge_ks(k,:));
end

if isnan(chunkSize)
    nChunk = 1;
    cell_chunks = 1:nCells;
else
    nChunk = ceil(nCells/chunkSize);
    cell_chunks = 1:(nChunk*chunkSize);
    cell_chunks = reshape(cell_chunks,chunkSize,[]);
    cell_chunks(~ismember(cell_chunks,1:nCells)) = NaN;
end

for perm_k = start_perm_k:nPermutes
    mdlParams.permuteInput = permute_idxs(perm_k,:);
    
    if perm_k > start_perm_k
        start_chunk_k = 1;
    elseif perm_k == start_perm_k
        start_chunk_k = find(any(cell_chunks==start_cell_k));
    end
    
    for chunk_k = start_chunk_k:nChunk
        chunk_cell_ks = cell_chunks(:,chunk_k);
        chunk_cell_ks = chunk_cell_ks(~isnan(chunk_cell_ks))';
        parfor k = chunk_cell_ks
            try
                [mdlResults{perm_k,k}, params{perm_k,k}] = all_call_coherence(batParams{k},inputs,'ridge',mdlParams);
            catch err
                disp(err)
                mdlResults{perm_k,k} = NaN;
                params{perm_k,k} = err;
            end
        end
        progress = 100*(((perm_k-1)*nChunk)+chunk_k)/(nPermutes*nChunk);
        fprintf('%d %% of cells processed\n',round(progress));
        toc(t);
        if ~isempty(output_folder)
            try
                lmResults = struct('mdlResults',{mdlResults},'params',{params},'inputs',{inputs},'permute_idxs',permute_idxs,'cell_ks',cell_ks);
                save(output_fname,'-v7.3','-struct','lmResults');
            catch err
                keyboard
            end
        end
    end
    
    
end
end