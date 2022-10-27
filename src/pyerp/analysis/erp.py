import sys
import os

import numpy as np
import mne
#import matplotlib.pyplot as plt

def export_epoch(data_dir,
                eeg_files,
                marker,
                file_type="vhdr",
                l_freq=0.1,
                h_freq=12,
                resample=None,
                tmin=-0.2,
                tmax=0.5,
                subject_code='sub01'):
    new_id = 2**16 # maximum id : 2147483647
    epochs = list()
    for idx, file in enumerate(eeg_files):
        raw = mne.io.read_raw(os.path.join(data_dir, file[0]+".%s"%file_type),
                            preload=True)
        raw.filter(l_freq, h_freq)
        if resample is not None:
            raw.resample(sfreq=resample)
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
            if event_type is None:
                continue
            tags = list()
            tags.append("subject:%s"%subject_code)
            tags.append("run:%s"%str(idx+1))
            tags.append("task:%s"%file[1])
            tags.append("trial:%s"%(None))
            tags.append("event:%s"%event_desc)
            tags.append(event_type)
            tags.append("marker:%s"%str(val))
            tags = '/'.join(tags)
            new_id += 1
            new_event_id[tags] = new_id
            events = mne.merge_events(events, [val], new_id)
        epochs.append(mne.Epochs(raw = raw,
                                events = events,
                                event_id = new_event_id,
                                tmin = tmin,
                                tmax = tmax))
    epochs = mne.concatenate_epochs(epochs)
    return epochs


def peak(epochs, r = 0.8, N = 10, mode = 'pos', ch = 'all', seed=None):
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

    if ch != 'all':
        epochs = epochs.pick_channels(ch)

    X = epochs.get_data()
    n_epochs, n_ch, n_samples = X.shape

    n_epochs_bootstrap = int(n_epochs*r)

    rng = np.random.default_rng(seed=seed)

    if mode == 'pos':
        func = np.max
    elif mode == 'neg':
        func = np.min

    amp = list()
    for reps in range(N):
        idx_epochs_shuffled = rng.integers(low=0, high=n_epochs, size=n_epochs_bootstrap)
        X_bootstrap = X[idx_epochs_shuffled,:,:]
        amp.append(func(X_bootstrap))
    
    amp = np.array(amp)

    return amp

def latency(T, nT, r = 0.8, N = 10, mode = 'pos', ch = 'all', seed=None, tmin=None, tmax='auto', alternative='two-sided', p_th=None):
    """
    returns latency with boot strapping.

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
        minimum time for latency. if set to 'auto', tmin will be set to 50 when mode is neg and 150 when mode is pos.
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

    X = dict()
    n_epochs = dict()
    n_epochs_bootstrap = dict()
    for stim in ['target','non-target']:
        X[stim] = epochs[stim].get_data()
        n_epochs[stim] = X[stim].shape[0]
        n_epochs_bootstrap[stim] = int(n_epochs[stim]*r)

    rng = np.random.default_rng(seed=seed)

    latency = list()
    for reps in range(N):
        idx_epochs_shuffled = dict()
        X_bootstrap = dict()
        for stim in ['target','non-target']:
            idx_epochs_shuffled[stim] = rng.integers(low=0, high=n_epochs[stim], size=n_epochs_bootstrap[stim])
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
            t_idx = np.where(p[idx]<=p_th)[0]
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
        latency.append(min(latency_tmp))
    
    return np.array(latency)


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

    r = np.mean(x1 ,axis=0) + np.mean(x2, axis=0)
    r = r / np.std(X, axis=0)
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


