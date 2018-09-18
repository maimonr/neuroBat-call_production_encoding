function [S,f] = calculate_lfp_spectrogram(csc,fs,baseline_lfp_samples)

if nargin > 2
    subtract_baseline_spectrum = true;
else
    subtract_baseline_spectrum = false;
end

Nchan = size(csc,1);
N = size(csc,2);

if subtract_baseline_spectrum
    winSize = size(baseline_lfp_samples,1);
    
    [fmin, fmax] = cwtfreqbounds(min(N,winSize),fs);
    nOctave = floor(log2(fmax/fmin));
else
    [fmin, fmax] = cwtfreqbounds(N,fs);
    nOctave = floor(log2(fmax/fmin));
end
n_voice_per_octave = 10;

[s,f] = cwt(csc(1,:),fs,'FrequencyLimits ',[fmin fmax],'VoicesPerOctave',n_voice_per_octave);

Nf = length(f);
S = nan(N,Nf,Nchan);
S(:,:,1) = abs(s');
for ch = 2:Nchan
    if ~any(isnan(csc))
        [s,f] = cwt(csc(ch,:),fs,'FrequencyLimits ',[fmin fmax],'VoicesPerOctave',n_voice_per_octave);
        S(:,:,ch) = abs(s');
    end
end

if subtract_baseline_spectrum
    
    baseline_range_idx = 1:winSize;
    n_baseline_samps = size(baseline_lfp_samples,2);
    S_baseline = nan(1,Nf,Nchan,n_baseline_samps);
    for b = 1:n_baseline_samps
        for ch = 1:Nchan
            csc_baseline = baseline_lfp_samples(baseline_range_idx,b,ch);
            cfs_baseline = abs(cwt(csc_baseline,fs,'NumOctaves ',nOctave,'VoicesPerOctave',n_voice_per_octave));
            S_baseline(1,:,ch,b) = nanmean(cfs_baseline,2);
        end
    end
    
    S_baseline = mean(S_baseline,4);
    
    S_baseline = repmat(S_baseline,N,1,1);
    S = S - S_baseline;
    
end
    
end