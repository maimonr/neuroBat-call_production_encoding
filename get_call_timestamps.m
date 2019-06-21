function [call_ts, cut_call_data, all_call_timestamps, included_call_times, included_call_ks] = get_call_timestamps(cut_call_data,stabilityBounds,params)

callpos = vertcat(cut_call_data.corrected_callpos);

usable_call_idx = callpos(:,1) - params.call_time_offset >= stabilityBounds(1) &...
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
    
    last_call_length = round(abs(diff(cut_call_data(included_call_ks{k}(end)).corrected_callpos)));
    call_start_idx = included_call_times{k}(1)-params.call_time_offset;
    call_end_idx = included_call_times{k}(end)+last_call_length+params.call_time_offset-1;
    all_call_timestamps{k} = call_start_idx:call_end_idx;
    k = k + 1;
end
call_ts = [all_call_timestamps{:}];

end