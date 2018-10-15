function [mdlResults, params, coherence_results, features] = all_call_coherence(batParams,inputs,mdlType,mdlParams)

add_yin_path = false;
if contains(getenv('HOSTNAME'),'ln') || contains(getenv('HOSTNAME'),'.savio') || contains(getenv('HOSTNAME'),'.brc')%savio Cluster
   savioFlag = true;
   addpath(genpath('/global/home/users/maimon/code/Myfunctions')); 
else
    savioFlag = false;
    if ~exist('yin','file')
        yinPath = genpath('C:\Program Files\MATLAB\R2018b\toolbox\yin\');
        add_yin_path = true;
        addpath(yinPath);
    end
end

if nargin < 4
    mdlParams = struct('onlyCoherence',false,'onlyFeats',false,'onlyStandardize',...
    false,'onlyNeuralData',false,'permuteInput',[]);
end
neural_data_type = 'spikes';
minCalls = 10;
nFilt = 200;
offset = nFilt/2;
smoothing_bin_size = 2;
spike_bin_size = 5;
smoothing_span_size = 150;
call_time_offset = 500;

cvType = 'fixed_ridge_k_cv';
banded_ridge_flag = true;

audio_fs = 250e3;
winSize = 1.5e-3;
overlap = winSize-1e-3;
nfft = 2^10;
max_f = 40e3;
envelope_window_length = 1e-3;
n_freq_bins = 6;
f_bounds = logspace(2,log10(max_f),n_freq_bins+1);

yin_wsize = round(winSize*audio_fs);

featureNames = {'f0','wEnt','RMS'};

yinParams = struct('thresh',0.01,'sr',audio_fs,'wsize',yin_wsize,...
    'hop',round(1e-3*audio_fs),'range',[],...
    'bufsize',[],'APthresh',2.5,...
    'maxf0',20000,'minf0',200);

pc_idx = 1:5; % Across all responsive cells the top 10 PCs capture >99% of explained variance for ~90% of cells, so nPC = 10 seems reasonable

params = struct('call_time_offset',call_time_offset,'nFilt',nFilt,...
    'spike_bin_size',spike_bin_size,'smoothing_span_size',smoothing_span_size,...
    'offset',offset,'smoothing_bin_size',smoothing_bin_size,'inputs',{inputs},...
    'mdlType',mdlType,'audio_fs',audio_fs,'envelope_window_length',envelope_window_length,...
    'envelope_ds_factor',[],'max_t',[],'neural_data_type',neural_data_type,...
    'minCalls',minCalls,'yinParams',yinParams,'bioacoustic_feature_names',{featureNames},...
    'pc_idx',pc_idx,'nBoot',0,'cvType',cvType,'mdlParams',mdlParams,'bandedRidge',banded_ridge_flag);

nInput = length(inputs);

cut_call_data = get_cut_call_data(batParams,params);

if isempty(cut_call_data)
    [mdlResults, params, coherence_results] = deal(NaN);
    return
end

[timestamps, neural_data, neural_params] = get_neural_data(batParams,params);
max_t = max(timestamps);
envelope_ds_factor = audio_fs/neural_params.fs;

params.max_t = max_t;
params.envelope_ds_factor = envelope_ds_factor;
params.specParams = struct('winSize',winSize,'overlap',overlap,'nfft',nfft,'f_bounds',f_bounds,'max_f',max_f);
neural_params.lfp_freq_band = [85 115];
params.neural_params = neural_params;

[call_ts, cut_call_data, all_call_timestamps, included_call_times,...
    included_call_ks] = get_call_timestamps(batParams,cut_call_data,params);

if ~neural_params.call_neural_common_reference
    call_ts = call_ts - timestamps(1);
    timestamps = timestamps - timestamps(1);
end

if isnan(call_ts)
    [mdlResults, params, coherence_results] = deal(NaN);
    return
end

binning_idx = slidingWin(length(call_ts),params.smoothing_bin_size,0);
call_wf = calculate_envelope(cut_call_data,all_call_timestamps,...
    included_call_times,included_call_ks,params);

if nargout > 2
    if strcmp(neural_data_type,'lfp')
        neural_data_for_coherence = neural_data(:,call_ts)';
    else
        neural_call_data = get_call_neural_data(timestamps,neural_data,call_ts,binning_idx,params);
        neural_data_for_coherence = neural_call_data;
    end
    [cxy,f] = mscohere(call_wf,neural_data_for_coherence,hamming(2^6),2^5,2^10,1e3);
    coherence_results = struct('coh',cxy,'f',f);
    if mdlParams.onlyCoherence
        mdlResults = NaN;
        return
    elseif strcmp(neural_data_type,'lfp')
        neural_call_data = get_call_neural_data(timestamps,neural_data,call_ts,binning_idx,params);
    end
else
    neural_call_data = get_call_neural_data(timestamps,neural_data,call_ts,binning_idx,params);
end

nChannel = size(neural_call_data,2);

call_feats = cell(1,nInput);
for input_k = 1:nInput
    call_feats{input_k} = get_feature(inputs{input_k},call_wf,cut_call_data,...
        all_call_timestamps,included_call_times,included_call_ks,params);
end

nPreds_per_feat = cellfun(@(x) size(x,1),call_feats);
nPreds = sum(nPreds_per_feat);
p = nPreds*nFilt;

if params.bandedRidge
    pred_idxs = cell(1,nInput);
    all_pred_k = 1;
    for input_k = 1:nInput
        input_pred_idx = zeros(nPreds_per_feat(input_k),nFilt);
        for pred_k = 1:nPreds_per_feat(input_k)
            input_pred_idx(pred_k,:) = all_pred_k:nPreds:p;
            all_pred_k = all_pred_k + 1;
        end
        pred_idxs{input_k} = reshape(input_pred_idx,1,[]);
    end
    
    assert(isempty(setdiff([pred_idxs{1},pred_idxs{2}],1:p)))
    assert(isempty(intersect(pred_idxs{1},pred_idxs{2})))
else
    pred_idxs = {1:p};
end


if ~isempty(mdlParams.permuteInput)
    permute_input_idx = find(mdlParams.permuteInput);
    for k = 1:length(permute_input_idx)
        input_k = permute_input_idx(k);
        feat = call_feats{input_k};
        permute_idx = randperm(size(feat,2));
        for feat_k = 1:size(feat,1)
            call_feats{input_k}(feat_k,:) = feat(feat_k,permute_idx);
        end
    end
end

call_feats = vertcat(call_feats{:});

if mdlParams.onlyFeats
    mdlResults = call_feats;
    [params, coherence_results] = deal(NaN);
    return
end

if smoothing_bin_size > 1
    
    call_feats_binned = zeros(size(call_feats,1),size(neural_call_data,1));
    for f_k = 1:size(call_feats,1)
        tmp = call_feats(f_k,:);
        call_feats_binned(f_k,:) = mean(tmp(binning_idx),2);
    end
else
    call_feats_binned = call_feats;
end

call_feats_padded = [zeros(size(call_feats,1),nFilt) call_feats_binned];

design_mat = zeros(length(neural_call_data),p);
for k = 1:length(neural_call_data)-offset
    design_mat(k,:) = reshape(call_feats_padded(:,k+offset:(nFilt + k - 1+offset)),1,[]);
end

design_mat = repmat(design_mat,nChannel,1);
neural_call_data = reshape(neural_call_data,1,[])';

if mdlParams.onlyStandardize
    mdlResults = struct('SDy',std(neural_call_data),'SDx',std(design_mat));
    [params, coherence_results] = deal(NaN);
    return
end

if nargout > 3
    features = design_mat; 
end

%%

switch mdlType
    case 'ridge'
            ridgeKs = logspace(3,7,5);%2.^(0.5:1:8);
            nCV = 5;
            nRidgeK = length(ridgeKs);
            nObservations = length(neural_call_data);
            mseCV = nan(1,nCV);
            R2CV = nan(1,nCV);
            rhoCV = nan(1,nCV);
            
            bCV = nan(p+1,nCV);
            b_scaled_CV = nan(p,nCV);
            
            logLikelihoodCV = nan(1,nCV);

            tic;
            switch params.cvType
                
                case 'nested_cv'
                    iteration_k = 1;
                    n_nest_cv = 5;
                    mseRidge = cell(1,n_nest_cv);
                    mseRidge_nested = cell(1,n_nest_cv);
                    nested_cv_idx = get_cv_idx(nObservations,nCV);
                    for nest_cv_k = 1:n_nest_cv
                        nested_testIdx = nested_cv_idx(1,nest_cv_k):nested_cv_idx(2,nest_cv_k)-1;
                            
                        nested_trainIdx = setdiff(1:nObservations,nested_testIdx);
                        nested_nObservations = length(nested_trainIdx);
                        cv_idx = get_cv_idx(nested_nObservations,nCV);
                        nested_design_mat = design_mat(nested_trainIdx,:);
                        
                        nested_yTrain = neural_call_data(nested_trainIdx);
                        mseRidge_cv = cell(1,nCV);
                        for cv_k = 1:nCV
                            testIdx = cv_idx(1,cv_k):cv_idx(2,cv_k)-1;
                            
                            trainIdx = setdiff(1:nested_nObservations,testIdx);
                            Xtest = nested_design_mat(testIdx,:);
                            yTrain = nested_yTrain(trainIdx);
                            yTest = nested_yTrain(testIdx);
                            [ZTZ,ZTy,muX_train,sigmaX_train] = get_ridge_mdl_inputs(nested_design_mat,trainIdx,yTrain);
                            [~,~,mseRidge_cv{cv_k}] = search_ridge_lambda(ridgeKs,pred_idxs,ZTZ,ZTy,sigmaX_train,muX_train,Xtest,yTest,yTrain);
                        end
                        t = toc;
                        fprintf('%d s elapsed, %d iterations\n',t,iteration_k);
                        iteration_k = iteration_k + 1;
                        
                        nested_yTest = neural_call_data(nested_testIdx);
                        nested_Xtest = design_mat(nested_testIdx,:);
                        
                        mseRidge_nested{nest_cv_k} = cat(nInput+1,mseRidge_cv{:});
                        mseRidge{nest_cv_k} = mean(mseRidge_nested{nest_cv_k},nInput+1);
                        [~,idx] = min(mseRidge{nest_cv_k}(:));
                        band_ridge_idx = cell(1,nInput);
                        [band_ridge_idx{1:nInput}] = ind2sub(repmat(nRidgeK,1,nInput),idx);
                        band_ridge_ks = ridgeKs([band_ridge_idx{:}]);
                        lambda_mat = get_lambda_mat(pred_idxs,band_ridge_ks,nInput,p);
                        
                        [ZTZ,ZTy,muX_train,sigmaX_train] = get_ridge_mdl_inputs(design_mat,nested_trainIdx,nested_yTrain);
                        [bCV(:,nest_cv_k),b_scaled_CV(:,nest_cv_k)] = fit_ridge_mdl(ZTZ,ZTy,nested_yTrain,muX_train,sigmaX_train,lambda_mat);
                        [mseCV(nest_cv_k),R2CV(nest_cv_k),rhoCV(nest_cv_k),logLikelihoodCV(nest_cv_k)] = assess_ridge_mdl_fit(bCV(:,nest_cv_k),nested_Xtest,nested_yTest);
                    end
                    
                case 'crossValidate'
                    mseRidge = cell(1,nCV);
                    mseRidge_nested = [];
                    cv_idx = get_cv_idx(nObservations,nCV);
                    for cv_k = 1:nCV
                        testIdx = cv_idx(1,cv_k):cv_idx(2,cv_k)-1;
                        
                        trainIdx = setdiff(1:nObservations,testIdx);
                        Xtest = design_mat(testIdx,:);
                        yTrain = neural_call_data(trainIdx);
                        yTest = neural_call_data(testIdx);
                        
                        [ZTZ,ZTy,muX_train,sigmaX_train] = get_ridge_mdl_inputs(design_mat,trainIdx,yTrain);
                        [~,~,mseRidge{cv_k}] = search_ridge_lambda(ridgeKs,pred_idxs,ZTZ,ZTy,sigmaX_train,muX_train,Xtest,yTest,yTrain);
                        
                    end
                    
                case 'fixed_ridge_k_cv'
                    mseRidge = cell(1,nCV);
                    mseRidge_nested = [];
                    cv_idx = get_cv_idx(nObservations,nCV);
                    for cv_k = 1:nCV
                        testIdx = cv_idx(1,cv_k):cv_idx(2,cv_k)-1;
                        
                        trainIdx = setdiff(1:nObservations,testIdx);
                        Xtest = design_mat(testIdx,:);
                        yTrain = neural_call_data(trainIdx);
                        yTest = neural_call_data(testIdx);
                        
                        band_ridge_ks = batParams.band_ridge_k;
                        
                        lambda_mat = get_lambda_mat(pred_idxs,band_ridge_ks,nInput,p);
                        
                        [ZTZ,ZTy,muX_train,sigmaX_train] = get_ridge_mdl_inputs(design_mat,trainIdx,yTrain);
                        b = fit_ridge_mdl(ZTZ,ZTy,neural_call_data,muX_train,sigmaX_train,lambda_mat);
                        [mseCV(cv_k),R2CV(cv_k),rhoCV(cv_k),logLikelihoodCV(cv_k)] = assess_ridge_mdl_fit(b,Xtest,yTest);
                        
                    end
                    
            end
        
        t = toc;
        %%
        
        n = length(neural_call_data);
        if ~contains(params.cvType,'fixed_ridge')
            mseRidge_avg = mean(cat(nInput+1,mseRidge{:}),nInput+1);
            [~,idx] = min(mseRidge_avg(:));
            band_ridge_idx = cell(1,nInput);
            [band_ridge_idx{1:nInput}] = ind2sub(repmat(nRidgeK,1,nInput),idx);
            band_ridge_idx = [band_ridge_idx{:}];
            band_ridge_ks = ridgeKs(band_ridge_idx);
        else
            band_ridge_idx = [];
            band_ridge_ks = batParams.band_ridge_k;
        end
        lambda_mat = get_lambda_mat(pred_idxs,band_ridge_ks,nInput,p);
        
        [ZTZ,ZTy,muX_train,sigmaX_train] = get_ridge_mdl_inputs(design_mat,1:n,neural_call_data);
        [b,b_scaled] = fit_ridge_mdl(ZTZ,ZTy,neural_call_data,muX_train,sigmaX_train,lambda_mat);
        [mse,R2,rho,logLikelihood,predSpikes] = assess_ridge_mdl_fit(b,design_mat,neural_call_data);
        [~, ~, sigmaX] = zscore(design_mat);
        % Using frequentist formulation of var see Cule et al. 2011
        % https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3228544/
        % test statistic: T(k) = b(k)/se(b(k)) 
        % Using variable names: T(k) = b(k)/posterior_std(k)
        % T ~ N(0,1) with large n
        nu = n - p;
        sig_sq_noise = ((neural_call_data - predSpikes)'*(neural_call_data - predSpikes))/nu;
        posterior_cov = sig_sq_noise*((ZTZ + lambda_mat)\(ZTZ/(ZTZ + lambda_mat)));
        
        % Alternate Bayesian formulation:
        %         sig_sq_noise = var(y - Z*b_scaled);
        %         posterior_cov = ((1/sig_sq_noise) * (Z'*Z + min_ridge_k*eye(size(Z,2))))^(-1);
        posterior_std = sqrt(diag(posterior_cov));
        posterior_std = posterior_std ./ sigmaX';
        
        sta_t = -offset*smoothing_bin_size:smoothing_bin_size:(offset-1)*smoothing_bin_size;
        %%
        mdlResults = struct('r2',R2,'b',b,'b_scaled',b_scaled,'rho',rho,...
            'sta_t', sta_t,'predSpikes',predSpikes,'min_ridge_k',band_ridge_ks,...
            'ridge_mse',{mseRidge},'mse',mse,'b_std',posterior_std,'ridgeK',ridgeKs,...
            'R2CV',R2CV,'min_mse_idx',band_ridge_idx,'logLikelihood',logLikelihood,...
            'rhoCV',rhoCV,'logLikelihoodCV',logLikelihoodCV,'mseCV',mseCV,...
            'mseRidge_nested',{mseRidge_nested});
end


if ~savioFlag && add_yin_path
    rmpath(yinPath)
end

end

function idx = get_cv_idx(nObservations,nCV)

cv_idx = round(linspace(1,nObservations+1,nCV+1));
cv_idx_reshape = vertcat(circshift(cv_idx,1),cv_idx);
idx = cv_idx_reshape(:,2:end);

end

function [ZTZ,ZTy,muX,sigmaX] = get_ridge_mdl_inputs(design_mat,trainIdx,y)

[Z, muX, sigmaX] = zscore(design_mat(trainIdx,:));
ZTZ = Z'*Z;
ZTy = (Z'*y);

end

function [R2,rho,mse,logLikelihood] = search_ridge_lambda(ridgeKs,pred_idxs,ZTZ,ZTy,sigmaX_train,muX_train,Xtest,yTest,yTrain)

nInput = length(pred_idxs);
nRidgeK = length(ridgeKs);
p = length(ZTy);
output_size = repmat(nRidgeK,1,nInput);
if numel(output_size) == 1
    output_size = [1 output_size];
end
mse = nan(output_size);
R2 = nan(output_size);
rho = nan(output_size);
logLikelihood = nan(output_size);
nIteration = prod(output_size(:));

[ridgeKMat{1:nInput}] = ndgrid(ridgeKs);

for k = 1:nIteration
    
    band_ridge_ks = cellfun(@(x) x(k),ridgeKMat);
    lambda_mat = get_lambda_mat(pred_idxs,band_ridge_ks,nInput,p);

    b = fit_ridge_mdl(ZTZ,ZTy,yTrain,muX_train,sigmaX_train,lambda_mat);
    [mse(k),R2(k),rho(k),logLikelihood(k)] = assess_ridge_mdl_fit(b,Xtest,yTest);

end

end

function lambda_mat = get_lambda_mat(pred_idxs,band_ridge_ks,nInput,p)

lambda_vec = ones(1,p);

for lambda_band_k = 1:nInput
    lambda_vec(pred_idxs{lambda_band_k}) = band_ridge_ks(lambda_band_k);
end

lambda_mat = diag(lambda_vec);

end

function [b,bScaled] = fit_ridge_mdl(ZTZ,ZTy,yTrain,muX_train,sigmaX_train,lambda_mat)

bScaled = (ZTZ + lambda_mat)\ZTy;
b = bScaled ./ sigmaX_train';
b = [mean(yTrain)-muX_train*b; b];

end

function [mse,R2,rho,logLikelihood,predSpikes] = assess_ridge_mdl_fit(b,Xtest,yTest)

predSpikes = b(1) + Xtest*b(2:end);

residuals = yTest - predSpikes ;
mse = mean(residuals.^2);
rss = mean((yTest - mean(yTest)).^2);
R2 = 1-mse/rss;
rho = corrcoef(predSpikes,yTest);
rho = rho(2);

n = length(yTest);
sigmaHat = sqrt(mse);
logLikelihood = -n*0.5*log(2*pi) - n*log(sigmaHat) - (1/(2*(sigmaHat^2)))*sum(residuals.^2);

end

function [timestamps, data, neural_params] = get_neural_data(batParams,params)

switch params.neural_data_type
    
    case 'spikes'
        
        fname = [batParams.batNum '_' batParams.cellInfo '.mat'];
        
        s = load([batParams.dataDir fname],'timestamps');
        timestamps = s.timestamps;
        data = [];
        neural_params.fs = 1e3;
        neural_params.call_neural_common_reference = true;
        
    case 'lfp'
        
        switch batParams.expType
            
            case 'juvenile'
                subtract_baseline = false;
                [baseDir, batNum, expDate] = get_data_dir(batParams,params);
                exp_date_str = datestr(expDate,'yyyymmdd');
                lfp_data_fname = [baseDir 'bat' batNum filesep 'neurologger_recording' exp_date_str '\lfpformat\LFP.mat'];
                
                s = load(lfp_data_fname,'timestamps','lfpData','orig_lfp_fs',...
                    'active_channels','baseline_lfp_samples','fs');
                
                timestamps = s.timestamps;
                lfpData = s.lfpData;
                active_channels = s.active_channels;
                if subtract_baseline
                    baseline_lfp_samples = s.baseline_lfp_samples;
                else
                    baseline_lfp_samples = [];
                end
                fs = s.fs;
                orig_lfp_fs = s.orig_lfp_fs;
                call_neural_common_reference = true;
              
            case 'adult'
                
                subtract_baseline = false;
                [~, batNum, expDate] = get_data_dir(batParams,params);
                
                lfp_data_dir = 'E:\ephys\adult_recording\lfp_data\';
                lfp_data_fname = [lfp_data_dir batNum '_' expDate '_LFP.mat'];
                
                s = load(lfp_data_fname,'timestamps_ms','voltage_traces',...
                    'indices_active_channels','sampling_freq'); 
                
                timestamps = 1e-3*s.timestamps_ms;
                lfpData = s.voltage_traces;
                active_channels = s.indices_active_channels;
                baseline_lfp_samples = [];
                fs = s.sampling_freq;
                orig_lfp_fs = [];
                call_neural_common_reference = false;
                
        end
        
        ch = batParams.lfp_channel_switch;
        
        if ischar(ch)
            switch ch
                case 'all'
                    ch_idx = true(1,size(lfpData,1));
                case 'tt'
                    nTetrode = 4;
                    n_channels_per_tetrode = 4;
                    tt =  batParams.lfp_tt;
                    
                    channels = reshape((0:(nTetrode*n_channels_per_tetrode)-1),nTetrode,n_channels_per_tetrode);
                    ch_idx = ismember(active_channels,channels(:,tt));
            end
        elseif isnumeric(ch)
            ch_idx = false(1,size(lfpData,1));
            ch_idx(ch) = true;
        end
        
        lfpData = lfpData(ch_idx,:);
        
        if fs < 1e3        
            upsample_scale = ceil(1e3/fs);
            timestamps_orig = timestamps;
            timestamps = upsample(timestamps,upsample_scale);
            timestamps(timestamps==0) = NaN;
            timestamps = fillmissing(timestamps,'linear');
            lfp_data_interp = zeros(sum(ch_idx),length(timestamps));
            for k = 1:sum(ch_idx)
               lfp_data_interp(k,:) = interp1(timestamps_orig,lfpData(k,:),timestamps); 
            end
            lfpData = lfp_data_interp;
        end
        
        [timestamps, idx] = unique(round(1e3*timestamps));
        
        idx = idx(timestamps>=0);
        timestamps = timestamps(timestamps>=0);
        data = lfpData(:,idx);
        
        if subtract_baseline
            
            baseline_timestamps = linspace(0,round(size(baseline_lfp_samples,1)/fs),size(baseline_lfp_samples,1));
            [~, idx] = unique(round(1e3*baseline_timestamps));
            idx = idx(1:(1e3*round(size(baseline_lfp_samples,1)/fs)));
            baseline_lfp_samples = baseline_lfp_samples(idx,:,:);
        end
        
        
        
        neural_params = struct('fs',1e3,'orig_fs',orig_lfp_fs,...
            'active_channels',active_channels,'subtract_baseline',subtract_baseline,...
            'baseline_lfp_samples',baseline_lfp_samples,...
            'call_neural_common_reference',call_neural_common_reference);
        
        
end


end

function neural_call_data = get_call_neural_data(timestamps,data,call_ts,binning_idx,params)

switch params.neural_data_type
    
    case 'spikes'
        
        edges = 0:params.spike_bin_size:max(timestamps)+1;
        interp_edges = 1:max(timestamps)+1;
        bin_centers = movmean(edges,2);
        bin_centers = bin_centers(2:end);
        binned_spikes = histcounts(timestamps,edges)/(1e-3*params.spike_bin_size);
        binned_spikes_interp = interp1(bin_centers,binned_spikes,interp_edges,[],'extrap');
        assert(~any(isnan(binned_spikes_interp)));
        call_spikes = binned_spikes_interp(call_ts);
        neural_call_data = smoothdata(call_spikes,2,'gaussian',params.smoothing_span_size);
        neural_call_data = mean(neural_call_data(binning_idx),2);
        
    case 'lfp'
        
        if params.neural_params.subtract_baseline
            [S,f] = calculate_lfp_spectrogram(data(:,call_ts),params.neural_params.fs,params.neural_params.baseline_lfp_samples);
        else
            [S,f] = calculate_lfp_spectrogram(data(:,call_ts),params.neural_params.fs);
        end
        neural_call_data = squeeze(mean(mean(S(:,(f>params.neural_params.lfp_freq_band(1) & f<params.neural_params.lfp_freq_band(2)),:),2),3));
        
        if size(binning_idx,2) > 1
            tmp_data = zeros(size(binning_idx,1),size(neural_call_data,2));
            for k = 1:size(neural_call_data,2)
                tmp = neural_call_data(:,k);
                tmp_data(:,k) = mean(tmp(binning_idx),2);
            end
            neural_call_data = tmp_data;
        end
        neural_call_data = zscore(neural_call_data);
end

end

function cut_call_data = get_cut_call_data(batParams,params)



switch batParams.expType
    case 'juvenile'
        
        s = load([batParams.dataDir batParams.batNum '_' batParams.cellInfo '.mat'],'cut_call_data');
        cut_call_data = s.cut_call_data;
        if all(isnan([cut_call_data.corrected_callpos]))
            cut_call_data = [];
            return
        end
        
    case 'adult'
        [baseDir, batNum, expDate] = get_data_dir(batParams,params);
        
        audioDir = [baseDir 'neurologger_recording' expDate '\audio\ch1\'];
        
        try
            s = load([audioDir 'cut_call_data.mat']);
        catch
           cut_call_data = []; 
           return
        end
        cut_call_data = s.cut_call_data;
        
        if isempty(cut_call_data)
            cut_call_data = [];
            return
        end
        
        cut_call_data = cut_call_data(~[cut_call_data.noise]);
        
        batIdx = unique(cellfun(@(call) find(cellfun(@(bNum) strcmp(bNum,batParams.batNum),call)),{cut_call_data.batNum}));
        bat_pair_nums = unique([cut_call_data.batNum]);
        
        if length(batIdx) == 1
            callpos = horzcat(cut_call_data.corrected_callpos);
            callpos = callpos(batIdx,:);
            [cut_call_data.corrected_callpos] = deal(callpos{:});
        else
            keyboard
        end
        
        call_info_loaded = false;
        
        for b_pair = 1:length(bat_pair_nums)
            call_info_fname = [audioDir 'call_info_' bat_pair_nums{b_pair} '_' batParams.call_echo '_' expDate '.mat'];
            if exist(call_info_fname,'file')
                s = load(call_info_fname);
                call_info = s.call_info;
                call_info_loaded = true;
            end
        end
        
        if ~call_info_loaded
            cut_call_data = [];
            return
        end
            
        
        assert(all([cut_call_data.uniqueID] == [call_info.callID]));
        
        bat_calls = cellfun(@(x) contains(x,batNum),{call_info.behaviors});
        cut_call_data = cut_call_data(bat_calls);
        
        
end

if length(cut_call_data) < params.minCalls
    cut_call_data = [];
    return
end

end

function [call_ts, cut_call_data, all_call_timestamps, included_call_times, included_call_ks] = get_call_timestamps(batParams,cut_call_data,params)

switch batParams.expType
    case 'juvenile'
        
        switch params.neural_data_type
            case 'spikes'
                s = load([batParams.dataDir batParams.batNum '_' batParams.cellInfo '.mat'],'stabilityBounds');
                stabilityBounds = s.stabilityBounds;
                
            case 'lfp'
                stabilityBounds = [-Inf Inf];
        end
        
    case 'adult'
        stabilityBounds = [-Inf Inf];
        
end

callpos = vertcat(cut_call_data.corrected_callpos);

usable_call_idx = callpos(:,1) - params.call_time_offset >=0 &...
    callpos(:,1) - params.call_time_offset >= stabilityBounds(1) &...
    callpos(:,2) + params.call_time_offset <= stabilityBounds(2) &...
    callpos(:,2) + params.call_time_offset <= params.max_t;

cut_call_data = cut_call_data(usable_call_idx);
callpos = callpos(usable_call_idx );

all_call_timestamps = cell(1,length(cut_call_data));
included_call_times = cell(1,length(cut_call_data));
included_call_ks = cell(1,length(cut_call_data));

k = 1;
call_k = 1;

while call_k <= length(cut_call_data)
    call_time = round(cut_call_data(call_k).corrected_callpos(1));
    
    included_call_times{k} = call_time;
    included_call_ks{k} = call_k;
    call_k = call_k + 1;
    while any((included_call_times{k}(end) + params.call_time_offset) >= callpos(call_k:end,1) - params.call_time_offset)
        call_time = round(cut_call_data(call_k).corrected_callpos(1));
        included_call_times{k} = [included_call_times{k} call_time];
        included_call_ks{k} = [included_call_ks{k} call_k];
        call_k = call_k + 1;
    end
    all_call_timestamps{k} = included_call_times{k}(1)-params.call_time_offset:included_call_times{k}(end)+params.call_time_offset-1;
    k = k + 1;
end
call_ts = [all_call_timestamps{:}];

end

function feature = get_feature(input,call_wf,cut_call_data,...
    all_call_timestamps,included_call_times,included_call_ks,params)



switch input
    
    case 'call_wf'
        
        feature = call_wf;
    case 'call_on'
        
        feature = call_wf~=mode(call_wf);
    case 'call_start'
        
        feature = diff([false (call_wf~=mode(call_wf))]) == 1;
    case 'call_end'
        
        feature = diff([false (call_wf~=mode(call_wf))]) == -1;
    case 'call_ps'
        
        feature = calculate_specs(cut_call_data,all_call_timestamps,included_call_times,included_call_ks,params);
        
    case 'call_ps_pca'
        
        pcaFlag = 'pca';
        feature = calculate_specs(cut_call_data,all_call_timestamps,included_call_times,included_call_ks,params,pcaFlag);

    case 'call_ps_pca_ortho'
        
        pcaFlag = 'pca_ortho';
        feature = calculate_specs(cut_call_data,all_call_timestamps,included_call_times,included_call_ks,params,pcaFlag);
        
    case 'bioacoustics'
        
        feature = calculate_bioacoustics(cut_call_data,all_call_timestamps,included_call_times,included_call_ks,params);

    case 'bioacoustics_ortho'
        
        orthoFlag = true;
        feature = calculate_bioacoustics(cut_call_data,all_call_timestamps,included_call_times,included_call_ks,params,orthoFlag);
end

end

function [features,featureNames] = calculate_bioacoustics(cut_call_data,all_call_timestamps,included_call_times,included_call_ks,params,varargin)

featureNames = params.bioacoustic_feature_names;
all_call_features = cell(1,length(all_call_timestamps));
allFeatureNames = {'f0','ap0','wEnt','specEnt','centroid','energyEnt','RMS'};

if ~isempty(varargin)
    orthoFlag = varargin{1};
else
    orthoFlag = false;
end

for k = 1:length(all_call_timestamps)
    all_call_features{k} = zeros(length(featureNames),length(all_call_timestamps{k}));
    for call_wf_k = 1:length(included_call_ks{k})
        cut = cut_call_data(included_call_ks{k}(call_wf_k)).cut';
        for feat_k = 1:length(featureNames)
            feature_idx = find(strcmp(featureNames{feat_k},allFeatureNames));
            idx = (params.call_time_offset + included_call_times{k}(call_wf_k) - included_call_times{k}(1));
            if feature_idx == 1 || feature_idx == 2
                
                nSamples = length(cut);
                params.yinParams.range = [1 nSamples];
                params.yinParams.bufsize = nSamples+2;
                
                L_frame = params.specParams.winSize*params.audio_fs;
                L_step = L_frame - params.specParams.overlap*params.audio_fs;
                nFrame = floor((nSamples-(L_frame-L_step))/L_step);
                
                call_wf_feats = zeros(2,nFrame);
                
                [f0,ap0] = calculate_yin_F0(cut',params.yinParams);
                
                if length(f0) < nFrame
                    f0 = padarray(f0,'NaN','post');
                    ap0 = padarray(ap0,'NaN','post');
                elseif length(f0) > nFrame
                    f0 = f0(1:nFrame);
                    ap0 = ap0(1:nFrame);
                end
                
                inf_idx = isinf(f0) | f0==0;
                
                f0(inf_idx) = NaN;
                ap0(inf_idx) = NaN;
                
                f0 = fillmissing(f0,'linear');
                ap0 = fillmissing(ap0,'linear');
                
                f0(f0<params.yinParams.minf0) = params.yinParams.minf0;
                f0(f0<params.yinParams.minf0) = params.yinParams.minf0;
                
                call_wf_feats(1,:) = log10(fillmissing(f0,'linear'));
                call_wf_feats(2,:) = fillmissing(ap0,'linear');
                call_wf_feats = call_wf_feats(feature_idx,:);
            else
                call_wf_feats = getCallFeatures(cut,params);
                call_wf_feats = call_wf_feats(feature_idx-2,:);
            end
            if any(~isreal(call_wf_feats(:)))
                keyboard
            end
            all_call_features{k}(feat_k,idx:idx+length(call_wf_feats)-1) = call_wf_feats;
        end
    end
end

features = [all_call_features{:}];

if orthoFlag
    for k = 1:size(features,1)
        call_on = features(k,:)~=mode(features(k,:));
        features(k,:) = features(k,:) - mean(features(k,call_on)).*call_on;
    end
end

end

function allFeats = getCallFeatures(callWF,params)

L = length(callWF);
L_frame = params.specParams.winSize*params.audio_fs;
L_step = L_frame - params.specParams.overlap*params.audio_fs;
nFrame = floor((L-(L_frame-L_step))/L_step);

nFeat = 5;

if nFrame > 0
    [weinerEntropy, spectralEntropy, centroid, energyEntropy, RMS] = deal(zeros(1,nFrame));
    for fr = 1:nFrame
        frameIdx = ((fr-1)*L_step +1):(((fr-1)*L_step)+L_frame);
        frame = callWF(frameIdx);
        [weinerEntropy(fr), spectralEntropy(fr), centroid(fr),...
            energyEntropy(fr), RMS(fr)] = getFeatures(frame,params);
    end
    allFeats = vertcat(weinerEntropy, spectralEntropy, centroid, energyEntropy, RMS);
else 
    allFeats = zeros(1,nFeat);
end


end
function [weinerEntropy, spectralEntropy, centroid, energyEntropy, RMS] = getFeatures(frame,params)

fs = params.audio_fs;

[AFFT, F] = afft(frame,fs);

%wiener entropy
weinerEntropy = exp(sum(log(AFFT)) ./ length(AFFT)) ./ mean(AFFT);

%spectral entropy
spectralEntropy = sentropy(AFFT);

%center of spectral mass (centroid)
centroid = sum( F' .* AFFT ) ./ sum(AFFT);

%Energy entropy
energyEntropy = sentropy(frame);

%RMS
RMS = sqrt(mean(frame.^2));

end
function [AFFT, F] = afft(sig,fs,nfft)

if size(sig,1) == 1
    sig = sig';
end

L = size(sig,1);

if nargin < 3 || isempty(nfft)
    nfft = 2^nextpow2(L);
end

F = fs/2*linspace(0,1,nfft/2+1);
AFFT = fft(sig,nfft)./L;
AFFT = 2*abs(AFFT(1:nfft/2+1,:));

end
function ent = sentropy(v)

if size(v,2) == 1
    v = v';
end

v = v + abs(min(min(v(:)),0));
v = v./(ones(size(v,1),1)*sum(v));
v(v==0) = 1;
ent = -sum(v.*log(v));

end
function [F0,ap] = calculate_yin_F0(callWF,P)

% Adapted from Vidush Mukund vidush_mukund@berkeley.edu
% March 13, 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function builds acts as a wrapper to call the YIN Algorithm package;
% based on the script written of Yosef Prat
%   P.minf0:    Hz - minimum expected F0 (default: 30 Hz)
%   P.maxf0:    Hz - maximum expected F0 (default: SR/(4*dsratio))
%   P.thresh:   threshold (default: 0.1)
%   P.relfag:   if ~0, thresh is relative to min of difference function (default: 1)
%   P.hop:      samples - interval between estimates (default: 32)
%   P.range:    samples - range of samples ([start stop]) to process
%   P.bufsize:  samples - size of computation buffer (default: 10000)
%	P.sr:		Hz - sampling rate (usually taken from file header)
%	P.wsize:	samples - integration window size (defaut: SR/minf0)
%	P.lpf:		Hz - intial low-pass filtering (default: SR/4)
%	P.shift		0: shift symmetric, 1: shift right, -1: shift left (default: 0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% call the yin function from the yin package
R = yin(callWF,P);

F0 = 2.^(R.f0 + log2(440));
ap = R.ap0;

if isempty(F0)
    F0 = 0;
    ap = 0;
end

F0 = smoothdata(F0,2);
ap = smoothdata(ap,2);

end

function call_wf = calculate_envelope(cut_call_data,all_call_timestamps,included_call_times,included_call_ks,params)

all_call_wf = cell(1,length(all_call_timestamps));
for k = 1:length(all_call_timestamps)
    
    all_call_wf{k} = zeros(1,length(all_call_timestamps{k}));
    
    for call_wf_k = 1:length(included_call_ks{k})
        cutEnv = envelope(cut_call_data(included_call_ks{k}(call_wf_k)).cut',params.audio_fs*params.envelope_window_length);
        cutEnv = downsample(cutEnv,params.envelope_ds_factor);
        idx = (params.call_time_offset + included_call_times{k}(call_wf_k) - included_call_times{k}(1));
        all_call_wf{k}(idx:idx+length(cutEnv)-1) = cutEnv;
    end
end

call_wf = [all_call_wf{:}];

end

function call_ps_feature = calculate_specs(cut_call_data,all_call_timestamps,included_call_times,included_call_ks,params,varargin)

if ~isempty(varargin)
    pcaFlag = varargin{1};
else
    pcaFlag = 'freq_bins';
end

specWin = kaiser(round(params.audio_fs*params.specParams.winSize),0.5);
all_call_ps = cell(1,length(all_call_timestamps));
freqs = linspace(1,params.specParams.max_f,params.specParams.nfft);

for k = 1:length(all_call_timestamps)
    all_call_ps{k} = zeros(params.specParams.nfft,length(all_call_timestamps{k}));
    for call_wf_k = 1:length(included_call_ks{k})
        cut = cut_call_data(included_call_ks{k}(call_wf_k)).cut';
        [~,~,~,ps] = spectrogram(cut,specWin,params.specParams.overlap*params.audio_fs,freqs,params.audio_fs);
        idx = (params.call_time_offset + included_call_times{k}(call_wf_k) - included_call_times{k}(1));
        all_call_ps{k}(:,idx:idx+size(ps,2)-1) = ps;
    end
end

call_ps = 10*log10([all_call_ps{:}]+eps);

switch pcaFlag
    case 'pca'
        [~,score] = pca(call_ps');
        
        call_ps_feature = score(:,params.pc_idx)';
        
    case 'pca_ortho'
        %         v = ones(1,params.specParams.nfft);
        %         v = v./norm(v);
        %         call_ps_ortho = call_ps - v'*(v*call_ps);
        
        call_ps_ortho = zeros(size(call_ps));
        call_on = call_ps(1,:)~=mode(call_ps(1,:));
        call_ps_ortho(:,call_on) = call_ps(:,call_on) - mean(call_ps(:,call_on),2);
        call_on = call_ps(1,:)~=mode(call_ps(1,:));
        coef = pca(call_ps_ortho(:,call_on)');
        score = coef*call_ps_ortho;
        
        call_ps_feature = score(params.pc_idx,:);
        
    case 'freq_bins'
        
        f_bounds_idx = false(length(params.specParams.f_bounds)-1 ,params.specParams.nfft);
        for f_k = 1:length(params.specParams.f_bounds) - 1
            [~,f_bounds_idx(f_k,:)] = inRange(freqs,[params.specParams.f_bounds(f_k) params.specParams.f_bounds(f_k+1)]);
        end
        
        f_bounds_idx = f_bounds_idx(sum(f_bounds_idx,2)>1,:);
        call_ps_feature = zeros(size(f_bounds_idx,1),size(call_ps,2));
        for f_k = 1:size(f_bounds_idx,1)
            call_ps_feature(f_k,:) = nanmean(call_ps(f_bounds_idx(f_k,:),:),1);
        end
end
end

function [baseDir, batNum, expDate, cellInfo] = get_data_dir(batParams,params)

switch params.neural_data_type
    
    case 'spikes'
        baseDir = batParams.baseDir;
        cellInfo = batParams.cellInfo;
        batNum = batParams.batNum;
        tetrodeStr = 'TT';
        expDate = cellInfo(1:strfind(cellInfo,tetrodeStr)-1);
        
    case 'lfp'
        
        batNum = batParams.batNum;
        expDate = batParams.expDate;
        
        baseDir = batParams.baseDir;
        
        cellInfo = [];
        
end
        
        
end