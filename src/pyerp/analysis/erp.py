import sys
import os

import numpy as np
import mne

from ..utils.analysis import get_target_from_trial, get_event_id_by_type
from .raw import split_raw_to_trial

def reject(epochs, eeg, eog = None):
    """
    Parameters
    ==========
    epochs : instance of mne.epochs
    min : minimum amplitude accepted. Unit is in micro Volts.
    max : maximum amplitude accepted. Unit is in micro Volts.
    """

    indices = [False for m in range(epochs.__len__())]

    if eeg is not None:
        eeg_data = epochs.get_data(picks = 'eeg', units = 'uV')
        for m in range(epochs.__len__()):
            max_eeg = np.max(np.max(np.squeeze(eeg_data[m,:,:])))
            min_eeg = np.min(np.min(np.squeeze(eeg_data[m,:,:])))
            if min_eeg < eeg[0] or max_eeg > eeg[1]:
                indices[m] = True

    if eog is not None: 
        eog_data = epochs.get_data(picks = 'eog', units = 'uV')
        for m in range(epochs.__len__()):
            max_eog = np.max(np.max(np.squeeze(eog_data[m,:,:])))
            min_eog = np.min(np.min(np.squeeze(eog_data[m,:,:])))
            if min_eog < eog[0] or max_eog > eog[1]:
                indices[m] = True
                
    epochs.drop(indices = indices)

    return epochs

def get_min_max(epochs, tags = None, average = False, picks = None, units = None, tmin = None, tmax = None):
    from ..utils.analysis import process_tags

    tags_processed = process_tags(tags)
    X = epochs[tags_processed].copy()
    if average:
        X = X.average()
    data = X.get_data(picks=picks, units=units, tmin=tmin, tmax=tmax)
    data = np.ravel(data)
    max = np.max(data)
    min = np.min(data)

    return min, max

def get_min_max_tnt(epochs, tags = None, picks = None, units = None, tmin = None, tmax = None):
    from ..utils.analysis import get_tags_binary
    
    tags_target, tags_nontarget = get_tags_binary(tags) 

    min_target, max_target = get_min_max(epochs, picks=picks, tags=tags_target, average=True, units=units, tmin=tmin, tmax=tmax)
    min_nontarget, max_nontarget = get_min_max(epochs, picks=picks, tags=tags_nontarget, average=True, units=units, tmin=tmin, tmax=tmax)

    max = np.max([max_target, max_nontarget])
    min = np.min([min_target, min_nontarget])

    return min, max

def get_min_max_tnt_task(epochs, tasks, picks = None, units = None, tmin = None, tmax = None):
    min = list()
    max = list()

    for task in tasks:
       tmp_min, tmp_max = get_min_max_tnt(epochs, tags = [task], picks = picks, units = units, tmin=tmin, tmax=tmax) 
       min.append(tmp_min)
       max.append(tmp_max)

    min = np.min(min)
    max = np.max(max)

    return min, max

def _epoch_from_raw(raw, marker, new_id_init, tmin, tmax, baseline, subject_code, task, run, trial = None):
    new_id = new_id_init
    events, event_id = mne.events_from_annotations(raw)
    new_event_id = dict()
    for key in list(event_id): 
        val = event_id[key]
        event_type = None
        event_desc = None
        for mrk in marker:
            if val == mrk[0]:
                event_type = mrk[1]
                event_desc = mrk[2]
                break
        if event_type is None or 'disable' in event_type.lower():
            continue
        tags = list()
        tags.append("subject:%s"%subject_code)
        tags.append("run:%d"%run)
        tags.append("task:%s"%task)
        tags.append("trial:%s"%(str(trial)))
        tags.append("event:%s"%event_desc)
        tags.append(event_type)
        tags.append("marker:%s"%str(val))
        tags = '/'.join(tags)
        new_id += 1
        new_event_id[tags] = new_id
        events = mne.merge_events(events, [val], new_id)
    epochs = mne.Epochs(raw = raw,
                        events = events,
                        event_id = new_event_id,
                        tmin = tmin,
                        tmax = tmax,
                        baseline = baseline)
    last_used_event_id = new_id
    return epochs, last_used_event_id


def export_epoch_from_raws(raws,
                          marker,
                          eog_channels = None,
                          filter = None,
                          zero_phase = True,
                          resample = None,
                          tmin = -0.2,
                          tmax = 0.5,
                          baseline = (None, 0),
                          subject_code = 'sub01',
                          split_trial = False,
                          apply_function = None,
                          ica = None,
                          **kwargs):
    new_id_init = 2**16 # maximum id : 2147483647
    #if ica_enable and ica_dir is None:
    #    ica_dir = data_dir

    epochs = list()
    for idx, raw_list in enumerate(raws):
        task = raw_list[1]
        run = idx + 1
        #raw = mne.io.read_raw(os.path.join(data_dir, fname), preload=True)
        raw = raw_list[0].copy()
        
        if eog_channels is not None:
            for ch in eog_channels:
                raw.set_channel_types({ch: 'eog'})

        #if ica_enable:
        #    ica = mne.preprocessing.read_ica(os.path.join(ica_dir, file[0] + "-%s-ica.fif"%ica_type))
        #    ica.apply(raw)
        if ica is not None:
            ica.apply(raw)
            
        if apply_function is not None:
            raw = apply_function(raw=raw, **kwargs)

        if filter is not None:
            if filter[0] == 'mne':
                raw.filter(filter[2][0], filter[2][1], phase=filter[1])
            elif filter[0] == 'sos':
                from .signal import apply_sosfilter
                raw.apply_function(apply_sosfilter, sos = filter[1], zero_phase = zero_phase, channel_wise = True, n_jobs = -1)
            elif filter[0] == 'ba':
                from .signal import apply_filter
                raw.apply_function(apply_filter, b = filter[1]['b'], a = filter[1]['a'], zero_phase = zero_phase, channel_wise = True, n_jobs = -1)

        if split_trial:
            marker_new_trial = get_event_id_by_type(marker, 'new-trial')
            raw_intervals = split_raw_to_trial(raw, marker_new_trial)

            marker_target = get_event_id_by_type(marker, 'target')
            for idx_trial, raw_interval in enumerate(raw_intervals):
                _epochs, last_used_event_id = _epoch_from_raw(raw_interval, marker, new_id_init, tmin, tmax, baseline, subject_code, task, run, idx_trial+1)
                new_id_init = last_used_event_id
                epochs.append(_epochs)
        else:
            _epochs, last_used_event_id = _epoch_from_raw(raw, marker, new_id_init, tmin, tmax, baseline, subject_code, task, run, None)
            new_id_init = last_used_event_id
            epochs.append(_epochs)
    epochs = mne.concatenate_epochs(epochs, add_offset=True) 
    if resample is not None:
        epochs.resample(sfreq=resample, n_jobs = -1)
    return epochs


def export_epoch(data_dir,
                eeg_files,
                marker,
                eog_channels = None,
                file_type = "vhdr",
                filter = None,
                zero_phase = True,
                resample = None,
                tmin = -0.2,
                tmax = 0.5,
                baseline = (None, 0),
                subject_code = 'sub01',
                split_trial = False,
                #ica_enable = False,
                #ica_dir = None,
                #ica_type = 'eog',
                ica = None, # by ica instance
                apply_function = None,
                **kwargs):
    new_id_init = 2**16 # maximum id : 2147483647
    #if ica_enable and ica_dir is None:
    #    ica_dir = data_dir

    epochs = list()
    for idx, file in enumerate(eeg_files):
        task = file[1]
        run = idx + 1
        if file_type is not None:
            fname = "%s.%s"%(file[0], file_type)
        else:
            fname = file[0]
        raw = mne.io.read_raw(os.path.join(data_dir, fname), preload=True)
        
        if eog_channels is not None:
            for ch in eog_channels:
                raw.set_channel_types({ch: 'eog'})

        #if ica_enable:
        #    ica = mne.preprocessing.read_ica(os.path.join(ica_dir, file[0] + "-%s-ica.fif"%ica_type))
        #    ica.apply(raw)
        if ica is not None:
            ica.apply(raw)
            
        if apply_function is not None:
            raw = apply_function(raw=raw, **kwargs)

        if filter is not None:
            if filter[0] == 'mne':
                raw.filter(filter[2][0], filter[2][1], phase=filter[1])
            elif filter[0] == 'sos':
                from .signal import apply_sosfilter
                raw.apply_function(apply_sosfilter, sos = filter[1], zero_phase = zero_phase, channel_wise = True, n_jobs = -1)
            elif filter[0] == 'ba':
                from .signal import apply_filter
                raw.apply_function(apply_filter, b = filter[1]['b'], a = filter[1]['a'], zero_phase = zero_phase, channel_wise = True, n_jobs = -1)

        if split_trial:
            marker_new_trial = get_event_id_by_type(marker, 'new-trial')
            raw_intervals = split_raw_to_trial(raw, marker_new_trial)

            marker_target = get_event_id_by_type(marker, 'target')
            for idx_trial, raw_interval in enumerate(raw_intervals):
                _epochs, last_used_event_id = _epoch_from_raw(raw_interval, marker, new_id_init, tmin, tmax, baseline, subject_code, task, run, idx_trial+1)
                new_id_init = last_used_event_id
                epochs.append(_epochs)
        else:
            _epochs, last_used_event_id = _epoch_from_raw(raw, marker, new_id_init, tmin, tmax, baseline, subject_code, task, run, None)
            new_id_init = last_used_event_id
            epochs.append(_epochs)
    epochs = mne.concatenate_epochs(epochs, add_offset=True) 
    if resample is not None:
        epochs.resample(sfreq=resample, n_jobs = -1)
    return epochs

def peak_latency(epochs, r = 0.8, N = 10, mode = 'pos', ch = 'all', units = None, seed=None):
    """
    returns peak amplitude with boot strapping.

    Parameters
    ==========
    epochs : array-like, shape of (n_epochs, n_ch, n_samples)
        epoch data.
    r : int, default = 0.8
        ratio of randomly chosen data samples. if set to 0.8, 80% of data will be taken.
    N : int, default = 10
        number of repetitions of bootstrap.
    mode : str ('pos' or 'neg'), default = 'pos'
        if set to 'pos', positive peak will be computed. 'neg' will be negative.

    References
    ==========
    Musso et al., Aphasia recovery by language training using a brain–computer interface: a proof-of-concept study, (2022)
    """
    from tqdm import tqdm
    
    time = epochs.times

    X = epochs.get_data(units = units, copy = True)
    n_epochs, n_ch, n_samples = X.shape
    
    ch_names = epochs.ch_names

    n_epochs_bootstrap = int(n_epochs*r)

    rng = np.random.default_rng(seed=seed)

    if mode == 'pos':
        func = np.max
        func_arg = np.argmax
    elif mode == 'neg':
        func = np.min
        func_arg = np.argmin

    amp = list()
    channels = list()
    latency = list()
    idx = np.arange(n_epochs)
    for reps in tqdm(range(N)):

        idx_epochs_shuffled = idx.copy()
        rng.shuffle(idx_epochs_shuffled)
        idx_epochs_shuffled = idx_epochs_shuffled[0:n_epochs_bootstrap]

        X_bootstrap = X[idx_epochs_shuffled,:,:]
        averaged = np.mean(X_bootstrap, axis = 0)
        amp.append(func(averaged))
        
        I = func_arg(func(averaged, axis = 0))
        latency.append(time[I])
        

        I = func_arg(func(averaged, axis = 1))
        channels.append(ch_names[I])
    
    amp = np.array(amp)
    latency = np.array(latency)

    return amp, latency, channels

def peak(epochs, r = 0.8, N = 10, mode = 'pos', ch = 'all', units = None, seed=None):
    """
    returns peak amplitude with boot strapping.

    Parameters
    ==========
    epochs : array-like, shape of (n_epochs, n_ch, n_samples)
        epoch data.
    r : int, default = 0.8
        ratio of randomly chosen data samples. if set to 0.8, 80% of data will be taken.
    N : int, default = 10
        number of repetitions of bootstrap.
    mode : str ('pos' or 'neg'), default = 'pos'
        if set to 'pos', positive peak will be computed. 'neg' will be negative.
    seed : None, int, ..., default=None
        seed for numpy.random.default_rng(), see detail on numpy documentation
    References
    ==========
    Musso et al., Aphasia recovery by language training using a brain–computer interface: a proof-of-concept study, (2022)
    """
    from tqdm import tqdm

    if ch != 'all':
        epochs = epochs.pick_channels(ch)

    X = epochs.get_data(units = units)
    n_epochs, n_ch, n_samples = X.shape
    
    ch_names = epochs.ch_names

    n_epochs_bootstrap = int(n_epochs*r)

    rng = np.random.default_rng(seed=seed)

    if mode == 'pos':
        func = np.max
        func_arg = np.argmax
    elif mode == 'neg':
        func = np.min
        func_arg = np.argmin

    amp = list()
    channels = list()
    idx = np.arange(n_epochs)
    for reps in tqdm(range(N)):

        idx_epochs_shuffled = idx.copy()
        rng.shuffle(idx_epochs_shuffled)
        idx_epochs_shuffled = idx_epochs_shuffled[0:n_epochs_bootstrap]

        X_bootstrap = X[idx_epochs_shuffled,:,:]
        averaged = np.mean(X_bootstrap, axis = 0)
        amp.append(func(averaged))

        I = func_arg(func(averaged, axis = 1))
        channels.append(ch_names[I])
    
    amp = np.array(amp)

    return amp, channels

def onset(T, nT, r = 0.8, N = 10, mode = 'pos', ch = 'all', seed=None, tmin=None, tmax='auto', alternative='two-sided', p_th=None):
    """
    returns onset obtained with boot strapping procedure.

    Parameters
    ==========
    T : array-like, shape of (n_epochs, n_ch, n_samples)
        epoch data of responses to target.
    nT : array-like, shape of (n_epochs, n_ch, n_samples)
        epoch data of responses to non-target.
    r : int, default = 0.8
        ratio of randomly chosen data samples. if set to 0.8, 80% of data will be taken.
    N : int, default = 10
        number of repetitions of bootstrap.
    mode : str ('pos' or 'neg'), default = 'pos'
        if set to 'pos', positive peak will be computed. 'neg' will be negative.
    ch : str, default='all'
        select the channels to detect latency.
    seed : None, int, ..., default=None
        seed for numpy.random.default_rng(), see detail on numpy documentation
    tmin : None, 'auto', float, default=None
        minimum time (in seconds) for latency. if set to 'auto', tmin will be set to 0.05 when mode is neg and 0.15 when mode is pos.
        if set to None, any latency could be taken.
    tmax : 'auto', float, None, default='auto'
        latency when significant difference was not detected., if 'auto' tmax will be returned, if None, None will be returned.
    alternative : 'two-sided' or 'one-sided', default = 'two-sided'
        type of t-test
    p_th : float, default = None
        threashold for detecting ERP response. if set None, it will be 0.05 when alternative is two-sided and 0.025 when alternative is one-sided

    References
    ==========
    Musso et al., Aphasia recovery by language training using a brain–computer interface: a proof-of-concept study, (2022)
    """
    from scipy import stats
    from tqdm import tqdm

    time = T.times

    if p_th is None:
        if alternative == 'one-sided':
            p_th = 0.025
        elif alternative == 'two-sided':
            p_th = 0.05

    if tmin == 'auto':
        if mode == 'pos':
            tmin = 0.15
        elif mode == 'neg':
            tmin = 0.05
        idx_min = np.where(time > tmin)[0][0]
    elif tmin == None:
        tmin = 0
        idx_min = 0
    else:
        idx_min = np.where(time > tmin)[0][0]

    if tmax == 'auto':
        tmax = time[-1]

    epochs = dict()
    epochs['target'] = T
    epochs['non-target'] = nT

    if ch != 'all':
        epochs['target'] = epochs['target'].pick_channels(ch)
        epochs['non-target'] = epochs['non-target'].pick_channels(ch)
        
    ch_names = epochs['target'].ch_names
    
    if ch_names != epochs['non-target'].ch_names:
        raise ValueError("channel subset for target and nontarget is not identical.")

    X = dict()
    n_epochs = dict()
    n_epochs_bootstrap = dict()
    for stim in ['target','non-target']:
        X[stim] = epochs[stim].get_data(copy = True)
        n_epochs[stim] = X[stim].shape[0]
        n_epochs_bootstrap[stim] = int(n_epochs[stim]*r)

    rng = np.random.default_rng(seed=seed)

    latency = list()
    channels = list()
    for reps in tqdm(range(N)):
        idx_epochs_shuffled = dict()
        X_bootstrap = dict()
        for stim in ['target','non-target']:

            idx = np.arange(n_epochs[stim])
            rng.shuffle(idx)
            idx_epochs_shuffled[stim] = idx[0:n_epochs_bootstrap[stim]]
            
            X_bootstrap[stim] = X[stim][idx_epochs_shuffled[stim],:,:]
        if alternative == 'one-sided':
            if mode == 'pos':
                t_score, p = stats.ttest_ind(X_bootstrap['target'], X_bootstrap['non-target'], axis=0, equal_var = False, alternative='greater')
            elif mode == 'neg':
                t_score, p = stats.ttest_ind(X_bootstrap['target'], X_bootstrap['non-target'], axis=0, equal_var = False, alternative='less')
        elif alternative == 'two-sided':
                t_score, p = stats.ttest_ind(X_bootstrap['target'], X_bootstrap['non-target'], axis=0, equal_var = False, alternative='two-sided')
        
        latency_tmp = list()
        for idx in range(p.shape[0]):
            t_idx = np.where(p[idx, :]<=p_th)[0]
            if len(t_idx) == 0:
                latency_tmp.append(tmax)
            else:
                if alternative == 'two-sided':
                    t_idx = t_idx[np.where(t_idx>=idx_min)[0]]
                    sign = np.mean(X_bootstrap['target'][:, idx, :], axis=0) - np.mean(X_bootstrap['non-target'][:, idx, :], axis=0)
                    if mode == 'pos':
                        t_idx = t_idx[np.where(sign[t_idx] >= 0)[0]]
                    elif mode == 'neg':
                        t_idx = t_idx[np.where(sign[t_idx] <= 0)[0]]
                elif alternative == 'one-sided':
                    t_idx = t_idx[np.where(t_idx>=idx_min)[0]]
                if len(t_idx) == 0:
                    latency_tmp.append(tmax)
                else:
                    latency_tmp.append(time[t_idx[0]])

        latency_tmp = np.array(latency_tmp) 
        I = np.argmin(latency_tmp)

        latency.append(np.min(latency_tmp))
        channels.append(ch_names[I])
    
    return np.array(latency), channels

def r_value(x1, x2):
    """
    returns r value.
    x1 - x2

    Parameters
    ==========
    x1 : array-like, shape of (n_epochs, n_ch, n_samples)
        epoch data of class 1, should be target
    x2 : array-like, shape of (n_epochs, n_ch, n_samples)
        epoch data of class 2, should be non-target
    """

    x1 = np.array(x1)
    x2 = np.array(x2)

    N1 = x1.shape[0]
    N2 = x2.shape[0]
    
    X = np.append(x1, x2, axis=0)

    r = np.mean(x1 ,axis=0) - np.mean(x2, axis=0)
    r = r / np.std(X, axis=0, ddof = 1)
    r = r * (np.sqrt(N1*N2)/(N1+N2))

    return r

def signed_square_r(x1, x2):
    """
    returns signed square r value.
    x1 - x2

    Parameters
    ==========
    x1 : array-like, shape of (n_epochs, n_ch, n_samples)
        epoch data of class 1, should be target
    x2 : array-like, shape of (n_epochs, n_ch, n_samples)
        epoch data of class 2, should be non-target
    """

    r = r_value(x1, x2)
    signed_r2 = r * np.absolute(r)

    return signed_r2


