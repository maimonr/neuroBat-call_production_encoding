function [call_wf, bat_ID] = calculate_all_call_envelope(cut_call_data,all_call_timestamps,included_call_times,included_call_ks,params)

[all_call_wf,all_call_bat_id] = deal(cell(1,length(all_call_timestamps)));
all_bat_nums = unique({cut_call_data([included_call_ks{:}]).batNum});

for k = 1:length(all_call_timestamps)
    
    [all_call_wf{k},all_call_bat_id{k}] = deal(zeros(1,length(all_call_timestamps{k})));
    
    for call_wf_k = 1:length(included_call_ks{k})
        cutEnv = envelope(cut_call_data(included_call_ks{k}(call_wf_k)).cut',params.audio_fs*params.envelope_window_length);
        cutEnv = downsample(cutEnv,params.envelope_ds_factor);
        
        batNum = cut_call_data(included_call_ks{k}(call_wf_k)).batNum;
        bat_id_num = find(strcmp(all_bat_nums,batNum));
        
        idx = 1 + (params.call_time_offset + included_call_times{k}(call_wf_k) - included_call_times{k}(1));
        all_call_wf{k}(idx:idx+length(cutEnv)-1) = cutEnv;
        all_call_bat_id{k}(idx:idx+length(cutEnv)-1) = bat_id_num*ones(size(cutEnv));
    end
end

call_wf = [all_call_wf{:}];
bat_ID = [all_call_bat_id{:}];

end