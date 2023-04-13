import scipy

def apply_filter(data, sos, zero_phase = True):
    # channel_wise should be set True, when it's called from mne_instance.apply_filter().
    if zero_phase:
        r = scipy.signal.sosfiltfilt(sos, data)
    else:
        r = scipy.signal.sosfilt(sos, data)
    return r