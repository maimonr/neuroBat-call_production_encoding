function neural_call_data = get_call_neural_data(timestamps,data,call_ts,binning_idx,params)

switch params.neural_data_type
    
    case 'spikes'
        
        edges = min(timestamps):params.spike_bin_size:max(timestamps)+1;
        interp_edges = min(timestamps):max(timestamps)+1;
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