function call_wf = calculate_all_call_envelope(cut_call_data,all_call_timestamps,included_call_times,included_call_ks,params)

all_call_wf = cell(1,length(all_call_timestamps));
for k = 1:length(all_call_timestamps)
    
    all_call_wf{k} = zeros(1,length(all_call_timestamps{k}));
    
    for call_wf_k = 1:length(included_call_ks{k})
        cutEnv = envelope(cut_call_data(included_call_ks{k}(call_wf_k)).cut',params.audio_fs*params.envelope_window_length);
        cutEnv = downsample(cutEnv,params.envelope_ds_factor);
        idx = 1 + (params.call_time_offset + included_call_times{k}(call_wf_k) - included_call_times{k}(1));
        all_call_wf{k}(idx:idx+length(cutEnv)-1) = cutEnv;
    end
end

call_wf = [all_call_wf{:}];

end